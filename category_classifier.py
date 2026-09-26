#!/usr/bin/env python3
"""
category_classifier.py - Universal Amazon-Style Auto-Categorization & Vision Engine

Provides:
  1. Standard Universal E-Commerce Taxonomy (Amazon top-level store departments):
     - Clothing, Shoes & Jewelry
     - Electronics & Accessories
     - Home & Kitchen
     - Beauty & Personal Care
     - Grocery & Gourmet Food
     - Sports & Outdoors
     - Tools & Home Improvement
     - Office Products
     - Musical Instruments
     - Toys & Games
  2. Local Offline AI Vision Classifier (MobileNetV3):
     - Classifies products by image pixels in ~30ms even if filename is an obscure code (e.g. IMG_3173, UUID, unsplash author tag).
     - Maps predicted ImageNet classes into canonical Amazon departments.
  3. Clean Title Refiner:
     - Detects non-descriptive camera raw filenames and replaces them with clean professional commercial titles.
  4. Perceptual reverse-image similarity neighbor inheritance.
"""

import os
import re
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import torch
import torchvision.models as models

import db
import image_search

# ---------------------------------------------------------------------------
# Amazon-Standard Universal E-Commerce Classification Taxonomy
# ---------------------------------------------------------------------------
TAXONOMY_MAP: List[Dict[str, Any]] = [
    {
        "category": "Clothing, Shoes & Jewelry",
        "icon": "shirt",
        "description": "Apparel, footwear, sneakers, designer bags, backpacks, luxury watches, and jewelry",
        "keywords": [
            # Footwear
            "sneaker", "sneakers", "shoe", "shoes", "boot", "boots", "sandal", "sandals", 
            "loafer", "loafers", "heel", "heels", "cleat", "cleats", "air jordan", "running shoe", 
            "footwear", "slide", "slides", "clog", "clogs", "slipper", "slippers", "oxford",
            # Bags & Luggage
            "backpack", "backpacks", "bag", "bags", "handbag", "handbags", "tote", "totes", 
            "purse", "purses", "luggage", "suitcase", "duffel", "satchel", "briefcase", 
            "fanny pack", "messenger bag", "crossbody", "wallet",
            # Apparel
            "shirt", "t-shirt", "tee", "hoodie", "jacket", "coat", "pants", "jeans", "shorts", 
            "dress", "skirt", "sweater", "apparel", "clothing", "garment", "suit", "blazer", 
            "hat", "cap", "beanie", "scarf", "glove", "gloves", "belt",
            # Watches & Jewelry
            "watch", "watches", "wristwatch", "wristwatches", "timepiece", "chronograph", 
            "rolex", "omega", "seiko", "casio", "patek", "smartwatch", "submariner",
            "ring", "necklace", "bracelet", "earring", "earrings", "jewelry", "pendant", 
            "jewel", "gemstone", "diamond", "cufflink", "brooch",
            # Eyewear
            "sunglasses", "sunglass", "shades", "glasses", "spectacles", "eyewear", "frames"
        ]
    },
    {
        "category": "Electronics & Accessories",
        "icon": "smartphone",
        "description": "Audio systems, headphones, computer peripherals, smart tech, cameras, and drones",
        "keywords": [
            "speaker", "speakers", "audio", "sound", "headphone", "headphones", "earbuds", 
            "earbud", "earphones", "headset", "airpods", "amplifier", "subwoofer", "microphone", 
            "soundbar", "sound system",
            "mouse", "mice", "keyboard", "keyboards", "trackpad", "webcam", "laptop", 
            "monitor", "display", "pc", "computer", "macbook", "usb", "hard drive", "ssd", 
            "dock", "mousepad", "peripheral",
            "phone", "smartphone", "iphone", "android", "tablet", "ipad", "camera", "dslr", 
            "drone", "smart device", "charger", "power bank", "gadget", "cables"
        ]
    },
    {
        "category": "Home & Kitchen",
        "icon": "home",
        "description": "Modern furniture, kitchenware, cookware, tableware, drinkware, lighting, and home care",
        "keywords": [
            # Furniture & Seating
            "chair", "chairs", "armchair", "sofa", "couch", "stool", "table", "desk", 
            "furniture", "bench", "seating", "cabinet", "shelf", "bookshelf", "bed", "dresser",
            # Kitchen & Cookware
            "teapot", "teapots", "kettle", "kettles", "tea maker", "tea set", "pan", "pot", 
            "cookware", "blender", "toaster", "chef knife", "kitchenware", "utensil", "cooker",
            # Tableware & Drinkware
            "mug", "mugs", "coffee mug", "cup", "cups", "glass", "glasses", "plate", "plates", 
            "bowl", "bowls", "tableware", "dining", "cutlery", "fork", "spoon", "knife",
            "water bottle", "waterbottle", "drink bottle", "flask", "thermos", "tumbler", 
            "tumblers", "canteen", "hydro flask", "stanley",
            # Lighting
            "lamp", "lamps", "light", "lights", "lighting", "lantern", "chandelier", "sconce", 
            "bulb", "led", "desk lamp", "floor lamp", "nightlight",
            # Home Care
            "iron", "steam iron", "steamer", "garment care", "vacuum", "dryer", "washer", 
            "laundry", "air purifier", "humidifier", "fan"
        ]
    },
    {
        "category": "Beauty & Personal Care",
        "icon": "sparkles",
        "description": "Skincare serums, luxury perfumes, cosmetics, fragrances, and personal grooming",
        "keywords": [
            "perfume", "perfumes", "fragrance", "fragrances", "cologne", "colognes", 
            "eau de parfum", "eau de toilette", "parfum", "scent", "aftershave", "body mist", "atomizer",
            "lotion", "lotions", "cream", "creams", "moisturizer", "serum", "serums", "serum bottle", "skincare", 
            "cosmetic", "cosmetics", "cleanser", "toner", "sunscreen", "lipstick", "mascara", 
            "foundation", "shampoo", "conditioner", "soap", "body wash", "beauty"
        ]
    },
    {
        "category": "Grocery & Gourmet Food",
        "icon": "coffee",
        "description": "Gourmet foods, artisanal dishes, beverages, snacks, and culinary ingredients",
        "keywords": [
            "food", "gourmet", "grocery", "dish", "meal", "biryani", "rice", "curry", "taco", 
            "tacos", "burrito", "kebab", "kebabs", "baklava", "pizza", "burger", "pasta", 
            "bread", "snack", "snacks", "dessert", "sweet", "sweets", "coffee", "tea", 
            "chocolate", "beverage", "drink", "spice", "spices", "sauce"
        ]
    },
    {
        "category": "Sports & Outdoors",
        "icon": "package",
        "description": "Fitness equipment, athletic sports gear, gym accessories, and outdoor recreation",
        "keywords": [
            "dumbbell", "weights", "yoga mat", "kettlebell", "barbell", "treadmill", 
            "basketball", "soccer", "football", "baseball", "tennis", "racket", "fitness", 
            "gym", "cycling", "bike", "helmet", "sports", "athletic", "outdoor", "camping"
        ]
    },
    {
        "category": "Tools & Home Improvement",
        "icon": "package",
        "description": "Power tools, cordless drills, hardware equipment, and home improvement fixtures",
        "keywords": [
            "drill", "drills", "hammer", "saw", "saws", "wrench", "screwdriver", "tool", 
            "tools", "power tool", "hardware", "pliers", "cordless drill", "impact driver", 
            "measuring tape", "toolbox"
        ]
    },
    {
        "category": "Office Products",
        "icon": "package",
        "description": "Desk organizers, fine pens, writing instruments, and professional office supplies",
        "keywords": [
            "pen", "pens", "fountain pen", "ballpoint", "pencil", "desk organizer", 
            "organizer", "stationery", "notebook", "binder", "paper", "stapler", "office"
        ]
    },
    {
        "category": "Musical Instruments",
        "icon": "sparkles",
        "description": "Guitars, acoustic and electric instruments, synthesizers, and music studio gear",
        "keywords": [
            "guitar", "electric guitar", "acoustic guitar", "piano", "keyboard piano", 
            "synthesizer", "drum", "drums", "violin", "trumpet", "flute", "ukulele", 
            "saxophone", "bass", "instrument", "musical instrument"
        ]
    },
    {
        "category": "Toys & Games",
        "icon": "package",
        "description": "Action figures, collectible toys, board games, and hobby model kits",
        "keywords": [
            "toy", "toys", "action figure", "figure", "figurine", "lego", "puzzle", 
            "board game", "doll", "plushie", "stuffed animal", "diecast", "model car", "collectible"
        ]
    }
]

# Mapping table from ImageNet keywords to Amazon-standard universal categories
VISION_IMAGENET_TO_CATEGORY = [
    # Clothing, Shoes & Jewelry
    (["loafer", "shoe", "sandal", "boot", "clog", "running shoe", "moccasin", "sock", 
      "sneaker", "cleat", "buckle", "jersey", "jean", "cardigan", "sweatshirt", "coat", 
      "suit", "trench coat", "vest", "kimono", "gown", "stole", "scarf", "bonnet", 
      "cowboy hat", "sombrero", "sunglasses", "sunglass", "spectacles", "watch", 
      "digital watch", "analog watch", "stopwatch", "backpack", "handbag", "purse", 
      "tote", "wallet", "mailbag", "umbrella"], "Clothing, Shoes & Jewelry"),
    
    # Electronics & Accessories
    (["laptop", "notebook", "mouse", "keyboard", "monitor", "screen", "television", 
      "loudspeaker", "speaker", "microphone", "headphone", "earphone", "cellular telephone", 
      "dial telephone", "ipod", "camera", "reflex camera", "camcorder", "projector", 
      "remote control", "joystick", "modem", "hard disc"], "Electronics & Accessories"),
    
    # Home & Kitchen
    (["chair", "armchair", "folding chair", "rocking chair", "desk", "table", "dining table", 
      "coffee table", "studio couch", "sofa", "wardrobe", "bookcase", "cabinet", "lamp", 
      "table lamp", "floor lamp", "lampshade", "teapot", "coffeepot", "kettle", "espresso maker", 
      "plate", "bowl", "consomme bowl", "soup bowl", "cup", "coffee mug", "goblet", 
      "wine bottle", "water bottle", "beer bottle", "pop bottle", "salt shaker", "pepper shaker", 
      "frying pan", "wok", "dutch oven", "pot", "pan", "toaster", "microwave", "refrigerator", 
      "stove", "washing machine", "vacuum", "iron", "pill bottle", "pitcher"], "Home & Kitchen"),
    
    # Beauty & Personal Care
    (["lotion", "sunscreen", "lipstick", "perfume", "cologne", "hair spray", "shampoo", 
      "soap dispenser", "cream", "cosmetic"], "Beauty & Personal Care"),
    
    # Grocery & Gourmet Food
    (["pizza", "hamburger", "hotdog", "burrito", "bagel", "bread", "french loaf", "pretzel", 
      "potpie", "ice cream", "espresso", "meat loaf", "carbonara", "cheeseburger"], "Grocery & Gourmet Food"),
    
    # Sports & Outdoors
    (["basketball", "soccer ball", "football", "tennis ball", "rugby ball", "baseball", 
      "volleyball", "golf ball", "dumbbell", "barbell", "bicycle", "racket", "ski", 
      "surfboard", "paddle"], "Sports & Outdoors"),
    
    # Tools & Home Improvement
    (["power drill", "drill", "hammer", "saw", "wrench", "screwdriver", "pliers", "plane", 
      "hatchet", "shovel", "axe"], "Tools & Home Improvement"),
    
    # Office Products
    (["ballpoint", "fountain pen", "quill", "rubber eraser", "pencil sharpener", "envelope", 
      "binder", "notebook"], "Office Products"),
    
    # Musical Instruments
    (["acoustic guitar", "electric guitar", "grand piano", "upright piano", "violin", "cello", 
      "trumpet", "trombone", "french horn", "saxophone", "flute", "drum", "drumstick", 
      "harmonica", "accordion", "banjo", "harp", "oboe"], "Musical Instruments"),
    
    # Toys & Games
    (["teddy", "teddy bear", "jigsaw puzzle", "toys", "lego", "balloon"], "Toys & Games")
]

# ---------------------------------------------------------------------------
# Global Singleton for Lightweight Vision Classifier
# ---------------------------------------------------------------------------
_vision_model = None
_vision_transforms = None
_vision_categories = None

def get_vision_classifier():
    """Initializes and returns the MobileNetV3 visual classifier (cached locally)."""
    global _vision_model, _vision_transforms, _vision_categories
    if _vision_model is None:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            _vision_model = models.mobilenet_v3_small(weights=weights).eval()
            _vision_transforms = weights.transforms()
            _vision_categories = weights.meta["categories"]
        except Exception as e:
            print(f"[Vision Classifier Init Error] {e}")
    return _vision_model, _vision_transforms, _vision_categories


def clean_text_tokens(text: str) -> List[str]:
    """Extracts lowercase alphanumeric words from a text string."""
    if not text:
        return []
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return [w for w in cleaned.split() if len(w) > 1]


def is_cryptic_or_raw_title(title: str, filename: Optional[str] = None) -> bool:
    """Detects whether a product name or filename is a non-descriptive camera code or random string."""
    clean = re.sub(r"[\s_\-]+[0-9]{7,}(?:[\s_\-][0-9]+)?", "", title).strip()
    raw = f"{clean} {filename or ''}".lower()

    # If it contains meaningful keywords from our taxonomy, it is definitely NOT cryptic!
    for item in TAXONOMY_MAP:
        for kw in item["keywords"]:
            if re.search(r"\b" + re.escape(kw.lower()) + r"\b", f" {clean.lower()} "):
                return False

    # Common camera / phone raw patterns
    if re.search(r"\b(img_\d+|dsc_\d+|screenshot|photo_\d+|image_\d+|pasted_image)\b", raw):
        return True
    
    # UUID-like pattern: 8-4-4-4-12 hex chars
    if re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}", raw):
        return True
        
    # Unsplash author-hash patterns (e.g. "irene kredenets kstsim1uvpw unsplash")
    if "unsplash" in raw or (len(clean.split()) >= 3 and any(len(part) > 8 for part in clean.split())):
        return True
        
    # Very short or pure number/test
    tokens = clean_text_tokens(clean)
    if not tokens or all(t.isdigit() for t in tokens):
        return True
        
    return False


def classify_image_with_vision(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Executes local offline visual classification on the image using MobileNetV3.
    Returns: {"category": str, "predicted_label": str, "confidence": float}
    """
    if not image_path or not os.path.exists(image_path):
        return None
        
    try:
        model, transforms, categories = get_vision_classifier()
        if not model or not transforms or not categories:
            return None
            
        with Image.open(image_path) as im:
            rgb_im = im.convert("RGB")
            tensor_in = transforms(rgb_im).unsqueeze(0)
            
        with torch.no_grad():
            output = model(tensor_in).squeeze(0).softmax(0)
            
        top_preds = output.topk(5)
        
        # Check top-5 predictions against vision-to-category mapping
        for prob, idx in zip(top_preds.values, top_preds.indices):
            label = categories[idx].lower()
            confidence = float(prob)
            
            for keyword_list, canonical_cat in VISION_IMAGENET_TO_CATEGORY:
                for kw in keyword_list:
                    if kw in label or label in kw:
                        return {
                            "category": canonical_cat,
                            "predicted_label": categories[idx],
                            "confidence": confidence
                        }
                        
        # Fallback to top-1 raw label if no direct keyword matched
        top1_idx = int(top_preds.indices[0])
        return {
            "category": "Home & Kitchen", # Sensible neutral fallback for physical goods
            "predicted_label": categories[top1_idx],
            "confidence": float(top_preds.values[0])
        }
    except Exception as e:
        print(f"[Vision Classification Error] {image_path}: {e}")
        return None


def match_text_against_taxonomy(text: str) -> Optional[Dict[str, Any]]:
    """Checks text against the standard Amazon taxonomy dictionary."""
    tokens = clean_text_tokens(text)
    joined_text = " " + " ".join(tokens) + " "

    best_match = None
    best_score = 0

    for item in TAXONOMY_MAP:
        score = 0
        for kw in item["keywords"]:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, joined_text):
                # Multi-word keyword matches get higher weight
                score += len(kw.split()) * 2

        if score > best_score:
            best_score = score
            best_match = item

    if best_score > 0 and best_match:
        return best_match
    return None


def find_matching_existing_category(
    target_category_name: str,
    existing_categories: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Searches existing database categories for exact or semantic match."""
    target_norm = target_category_name.lower().strip()

    # Exact match first
    for c in existing_categories:
        if c["name"].lower().strip() == target_norm:
            return c

    # Substring containment match (e.g. "Footwear" inside "Footwear & Sneakers")
    for c in existing_categories:
        c_norm = c["name"].lower().strip()
        if (len(c_norm) >= 5 and c_norm in target_norm) or (len(target_norm) >= 5 and target_norm in c_norm):
            return c

    return None


def refine_product_title(
    raw_title: str, 
    raw_image_path: Optional[str] = None, 
    filename: Optional[str] = None,
    predicted_label: Optional[str] = None
) -> str:
    """
    Produces a clean, commercial title for the product.
    If the name is a cryptic raw code (e.g. IMG_3173, Unsplash hash), generates a professional title.
    """
    # Strip timestamp / counter suffixes (e.g. _1789319907 or 1789223502_1)
    clean_base = re.sub(r"[\s_\-]+[0-9]{7,}(?:[\s_\-][0-9]+)?", "", raw_title)
    clean_base = re.sub(r"[_\-\+]+", " ", clean_base).strip().title()
    
    if not is_cryptic_or_raw_title(clean_base, filename):
        return clean_base
        
    # If vision detected a recognizable item
    if predicted_label:
        label_clean = re.sub(r"[,_]+", " ", predicted_label).strip().title()
        return f"Classic {label_clean}"
        
    if raw_image_path and os.path.exists(raw_image_path):
        vision_res = classify_image_with_vision(raw_image_path)
        if vision_res and vision_res.get("predicted_label"):
            label_clean = re.sub(r"[,_]+", " ", vision_res["predicted_label"]).strip().title()
            return f"Classic {label_clean}"
            
    return clean_base if len(clean_base) > 2 else "Commercial Product"


def resolve_product_category(
    product_name: str,
    raw_image_path: Optional[str] = None,
    tags: str = "",
    description: str = "",
    filename: Optional[str] = None
) -> Tuple[int, str, bool]:
    """
    Determines the best Amazon-standard category for a product.
    
    Priority:
      1. If title/filename is descriptive, match against Amazon Taxonomy keywords.
      2. If title is cryptic/generic, or keywords do not match, run AI Vision Classifier.
      3. Reverse visual similarity check against existing catalog items.
      4. Links to or registers the category in SQLite.
      
    Returns: (category_id, category_name, is_newly_created)
    """
    existing_cats = db.list_categories()
    combined_text = f"{product_name} {tags} {description} {os.path.splitext(filename or '')[0]}"
    
    matched_cat_name = None
    matched_tax_entry = None
    
    is_cryptic = is_cryptic_or_raw_title(product_name, filename)
    
    # 1. If descriptive, try semantic text matching first
    if not is_cryptic:
        matched_tax_entry = match_text_against_taxonomy(combined_text)
        if matched_tax_entry:
            matched_cat_name = matched_tax_entry["category"]

    # 2. If no text match or title is cryptic, run local AI Vision Classifier
    if not matched_cat_name and raw_image_path and os.path.exists(raw_image_path):
        vision_res = classify_image_with_vision(raw_image_path)
        if vision_res:
            matched_cat_name = vision_res["category"]

    # 3. If still not resolved, try reverse visual search against catalog
    if not matched_cat_name and raw_image_path and os.path.exists(raw_image_path):
        similar_items = image_search.reverse_image_search(raw_image_path, top_k=3)
        for sim in similar_items:
            if sim.get("similarity_score", 0) >= 0.80 and sim.get("category_name"):
                matched_cat_name = sim["category_name"]
                break

    # 4. Fallback text match if vision didn't resolve
    if not matched_cat_name:
        matched_tax_entry = match_text_against_taxonomy(combined_text)
        if matched_tax_entry:
            matched_cat_name = matched_tax_entry["category"]

    # 5. Default Amazon Category if completely unknown
    if not matched_cat_name:
        matched_cat_name = "Home & Kitchen"

    # Find metadata for matched category
    tax_meta = next((item for item in TAXONOMY_MAP if item["category"] == matched_cat_name), None)
    icon = tax_meta["icon"] if tax_meta else "package"
    desc = tax_meta["description"] if tax_meta else "Universal E-Commerce Catalog"

    # Resolve against SQLite database
    matched_existing = find_matching_existing_category(matched_cat_name, existing_cats)
    if matched_existing:
        return matched_existing["id"], matched_existing["name"], False
    else:
        new_cat = db.create_category(
            name=matched_cat_name,
            description=desc,
            icon=icon
        )
        return new_cat["id"], new_cat["name"], True
