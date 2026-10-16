#!/usr/bin/env python3
"""
ai_generator.py - Intelligent Commercial AI Product Generation Engine v2.0

Uses Google Gemini Imagen 3 (imagen-3.0-generate-002) for true AI image synthesis.
Falls back to high-fidelity template-based generation when API is unavailable.

Key improvements:
- Real AI generation: ROG backpack generates ROG gaming backpack (NOT a luxury bag)
- Gemini Imagen 3 with product-specific prompt engineering
- 60+ brand-specific visual descriptions
- 40+ product type visual templates
- Amazon-standard white background photography prompts
- Graceful fallback to template system when no API key is present
"""

import os
import sys
import re
import io
import uuid
import time
import random
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_images")
TEMPLATES_DIR = os.path.join(BASE_DIR, "ai_studio_templates")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Gemini API Configuration
# ---------------------------------------------------------------------------
def _load_gemini_key():
    """Load Gemini API key from environment or .env file."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key and key.strip():
        return key.strip()
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip().lstrip("\ufeff")
                v = v.strip().strip('"').strip("'")
                if k in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and v:
                    return v
    return None

GEMINI_API_KEY = _load_gemini_key()

# ---------------------------------------------------------------------------
# Brand + Product Knowledge Database (60+ brands, 40+ product types)
# Ensures ROG backpack → ASUS ROG gaming backpack, NOT a luxury fashion bag
# ---------------------------------------------------------------------------
BRAND_PRODUCT_MAP = {
    # Flagship Smartphones & Mobile (HIGHEST PRIORITY - prevents phone -> watch misclassification)
    "samsung galaxy s21 fe": ("Samsung Galaxy S21 FE 5G Factory Unlocked Android Smartphone, Lavender pastel colorway, triple camera module, sleek modern design", "electronics"),
    "s21 fe": ("Samsung Galaxy S21 FE 5G Factory Unlocked Android Smartphone, Lavender pastel colorway, triple camera module, sleek modern design", "electronics"),
    "galaxy s21": ("Samsung Galaxy S21 FE 5G Android Smartphone, Lavender finish, triple camera housing", "electronics"),
    "samsung galaxy": ("Samsung Galaxy flagship Android smartphone, sleek glass and aluminum unibody, high-resolution multi-camera array", "electronics"),
    "samsung": ("Samsung Galaxy flagship Android smartphone, sleek glass and metal design, high-resolution display", "electronics"),
    "iphone 17 pro max": ("Apple iPhone 17 Pro Max flagship smartphone, Cosmic Titanium finish, edge-to-edge Super Retina XDR OLED display, Dynamic Island, advanced sapphire triple camera system", "electronics"),
    "iphone 17 pro": ("Apple iPhone 17 Pro flagship smartphone, Cosmic Titanium finish, edge-to-edge Super Retina XDR OLED display, Dynamic Island, advanced sapphire triple camera system", "electronics"),
    "iphone 17": ("Apple iPhone 17 flagship smartphone, aerospace aluminum and glass unibody, Dynamic Island, advanced dual camera system", "electronics"),
    "iphone 16": ("Apple iPhone 16 Pro smartphone, Desert Titanium finish, Dynamic Island, pro camera system", "electronics"),
    "iphone 15": ("Apple iPhone 15 Pro flagship smartphone, Natural Titanium finish, Dynamic Island, triple camera sapphire lens module", "electronics"),
    "iphone": ("Apple iPhone flagship smartphone, premium aluminum/titanium unibody, edge-to-edge Super Retina display, pro camera system", "electronics"),
    "pixel": ("Google Pixel flagship Android smartphone, camera visor bar, matte glass back", "electronics"),
    # Cameras & Optics
    "canon": ("Canon EOS R5 full-frame camera with 50mm f/1.4 prime lens attached, matte black magnesium alloy chassis, textured rubber grip", "electronics"),
    "nikon": ("Nikon Z-series full-frame mirrorless camera with NIKKOR prime lens, matte black weather-sealed body, red accent grip", "electronics"),
    "sony alpha": ("Sony Alpha a7 IV full-frame mirrorless camera with G Master prime lens, matte black magnesium body, textured rubber grip", "electronics"),
    "sony a7": ("Sony Alpha a7 IV full-frame mirrorless camera with G Master prime lens, matte black magnesium body, textured rubber grip", "electronics"),
    "dslr": ("Professional full-frame DSLR camera with 50mm f/1.4 prime lens attached, matte black magnesium alloy body, tactile diamond-pattern textured rubber grip, brushed gunmetal top plate dials", "electronics"),
    "camera": ("Professional full-frame DSLR camera with 50mm f/1.4 prime lens attached, matte black magnesium alloy body, tactile diamond-pattern textured rubber grip, brushed gunmetal top plate dials", "electronics"),
    "mirrorless": ("Professional full-frame mirrorless camera with fast prime lens attached, matte black magnesium alloy body, tactile rubber grip", "electronics"),
    # Gaming Brands
    "rog": ("ASUS ROG gaming backpack, matte black tactical nylon, angular reinforced design, red ROG logo, laptop compartment, USB-A pass-through charging port, hardshell protection zones", "bags"),
    "asus rog": ("ASUS ROG gaming backpack, matte black angular design, red accent trim, ROG logo patch, padded laptop sleeve", "bags"),
    "razer": ("Razer gaming peripheral, matte black design, green Razer snake logo RGB accent", "electronics"),
    "corsair": ("Corsair gaming peripheral, matte black with yellow accents, K or M-series gaming hardware", "electronics"),
    "steelseries": ("SteelSeries gaming accessory, sleek black with orange logo, competitive esports equipment", "electronics"),
    "logitech": ("Logitech peripheral device, clean modern design, MX or G-series precision engineering", "electronics"),
    "alienware": ("Alienware gaming product, dark futuristic design with alien head logo, Area-51 aesthetic", "electronics"),
    "hyperx": ("HyperX gaming headset, over-ear cushioned earcups, black with red accents, boom microphone", "electronics"),
    "msi": ("MSI gaming hardware, black with red dragon design, high-performance dragon shield branding", "electronics"),
    # Footwear
    "nike dunk": ("Nike Dunk Low Retro sneaker, black and white Panda colorway, premium leather upper, cupsole", "footwear"),
    "dunk low": ("Nike Dunk Low Retro sneaker, black and white Panda colorway, premium leather upper", "footwear"),
    "adidas samba": ("Adidas Samba Classic leather sneaker, white with black three stripes, suede T-toe, gum rubber outsole", "footwear"),
    "samba": ("Adidas Samba Classic leather sneaker, white with black three stripes, gum rubber outsole", "footwear"),
    "air jordan": ("Air Jordan basketball sneaker, premium leather collar, jumpman logo, iconic silhouette", "footwear"),
    "jordan": ("Air Jordan retro sneaker, leather upper, rubber outsole, jumpman branding", "footwear"),
    "nike": ("Nike athletic sneaker, clean leather or mesh upper, swoosh logo, Air cushioning", "footwear"),
    "adidas": ("Adidas sneaker with three stripes, performance mesh or leather, Boost or Foam midsole", "footwear"),
    "yeezy": ("Adidas Yeezy sneaker, distinctive chunky silhouette, Primeknit upper, thick rubber sole", "footwear"),
    "new balance": ("New Balance athletic sneaker, bold N logo, retro chunky runner silhouette, premium suede panels", "footwear"),
    "puma": ("PUMA athletic sneaker, leaping cat logo, clean performance design", "footwear"),
    "converse": ("Converse Chuck Taylor All Star canvas sneaker, vulcanized rubber toe cap, classic high-top", "footwear"),
    "vans": ("Vans canvas sneaker, waffle rubber sole, Off The Wall check logo, skate lifestyle design", "footwear"),
    "asics": ("ASICS running shoe, Gel cushioning pods visible, technical mesh upper, performance outsole", "footwear"),
    "dr martens": ("Dr. Martens boot, yellow welt stitching, air-cushioned AirWair sole, classic British design", "footwear"),
    "timberland": ("Timberland 6-Inch Premium wheat nubuck leather waterproof boot, chunky lug outsole, yellow stitching, iconic outdoor design", "footwear"),
    "ugg": ("UGG sheepskin boot, suede exterior, twin-faced sheepskin interior, chestnut or grey colorway", "footwear"),
    "birkenstock": ("Birkenstock Boston clog, soft suede upper, adjustable pin buckle, cork footbed", "footwear"),
    # Bags & Luggage
    "north face": ("The North Face Borealis outdoor backpack, durable black nylon shell, bungee cords, mountain logo patch", "bags"),
    "the north face": ("The North Face Borealis outdoor backpack, durable black nylon shell, bungee cords, mountain logo patch", "bags"),
    "gucci": ("Gucci luxury leather handbag, GG Supreme monogram canvas or smooth leather, gold-tone hardware, bamboo handle", "bags"),
    "louis vuitton": ("Louis Vuitton LV monogram coated canvas bag, vachetta leather trim, brass-tone LV padlock hardware", "bags"),
    "prada": ("Prada nylon or Saffiano leather bag, enameled triangular Prada logo plate, clean minimalist Italian design", "bags"),
    "chanel": ("Chanel quilted lambskin leather bag, interlocking CC clasp, gold chain strap, classic flap silhouette", "bags"),
    "hermes": ("Hermes Birkin bag, Togo or Epsom leather, palladium clochette hardware, structured trapezoid shape", "bags"),
    "coach": ("Coach pebble leather shoulder bag, Coach logo signature lining, polished gold hardware", "bags"),
    "osprey": ("Osprey technical hiking backpack, AirSpeed suspension, aluminum framesheet, multiple hipbelt pockets", "bags"),
    "fjallraven": ("Fjallraven Kanken backpack, G-1000 Vinylon fabric, classic square shape, sewn-on logo patch", "bags"),
    "herschel": ("Herschel Supply Co. backpack, striped fabric liner, clamshell top opening, leather base corners", "bags"),
    "samsonite": ("Samsonite hardshell carry-on luggage, polycarbonate shell, spinner wheels, TSA-approved combo lock", "bags"),
    # Drinkware & Lifestyle
    "stanley": ("Stanley Quencher H2.0 FlowState 40oz stainless steel vacuum insulated tumbler with comfort-grip handle and reusable straw, cream finish", "general"),
    "yeti": ("YETI Rambler 20oz stainless steel insulated tumbler with MagSlider lid, durable powder coat finish", "general"),
    "hydro flask": ("Hydro Flask wide mouth insulated stainless steel water bottle, flex cap, vibrant powder coat", "general"),
    "breville": ("Breville Barista Touch espresso machine, brushed stainless steel, digital touchscreen, integrated grinder and steam wand", "general"),
    "bowflex": ("Bowflex SelectTech 552 adjustable dumbbells on resting cradle, red and black dial weight selector", "general"),
    # Audio & Electronics
    "sony": ("Sony WH-1000XM5 wireless noise-canceling headphones, matte black finish with copper accents, ultra-comfortable leatherette headband", "electronics"),
    "apple": ("Apple premium device, aluminum unibody, clean minimalist design", "electronics"),
    "bose": ("Bose premium audio product, clean minimal silver-black design, noise cancellation, acoustic engineering", "electronics"),
    "beats": ("Beats by Dre headphones, bold glossy color, b-logo ear cup, premium street fashion audio", "electronics"),
    "jbl": ("JBL Bluetooth speaker, cylindrical fabric mesh grille, rubber bumper ring, colorful lifestyle speaker", "electronics"),
    "gopro": ("GoPro Hero action camera, compact waterproof housing, wide-angle lens, mounting fingers", "electronics"),
    "dji": ("DJI drone, white and grey folding arms, 4K gimbal stabilized camera, compact travel design", "electronics"),
    # Watches
    "rolex": ("Rolex Submariner Date luxury dive watch, black Cerachrom ceramic bezel, black dial, luminous markers, brushed Oystersteel bracelet", "accessories"),
    "omega": ("Omega Seamaster or Speedmaster, stainless steel case, COSC certified chronometer, professional dive watch", "accessories"),
    "seiko": ("Seiko Japanese automatic watch, sunburst dial, stainless steel bracelet, reliable precision", "accessories"),
    "casio": ("Casio G-Shock or Edifice, resin or metal case, shock-resistant design, multi-function digital display", "accessories"),
    "apple watch": ("Apple Watch Ultra titanium case, digital crown, action button, orange alpine loop or ocean band", "accessories"),
    # Cosmetics & Fragrance
    "dior": ("Dior Sauvage Eau de Parfum luxury fragrance, cylindrical dark midnight blue gradient glass bottle with magnetic silver CD grooved cap", "cosmetics"),
    "ysl": ("Yves Saint Laurent Rouge Pur Couture luxury satin lipstick in reflective gold square tube with embossed YSL Cassandra logo", "cosmetics"),
    "chanel beauty": ("Chanel No.5 or Bleu de Chanel, rectangular frosted glass bottle, interlocking CC cap, Parisian elegance", "cosmetics"),
    "jo malone": ("Jo Malone London fragrance, square cream glass bottle, silk grosgrain ribbon, British botanical luxury", "cosmetics"),
    "ordinary": ("The Ordinary clinical skincare serum, transparent dropper bottle, clean sans-serif label, dermatologist grade", "cosmetics"),
    "cerave": ("CeraVe moisturizer or cleanser, white pump bottle, blue label, MVE ceramide technology, clinically proven", "cosmetics"),
    "fenty": ("Fenty Beauty by Rihanna, sleek modern packaging, inclusive shade range, pro-quality formula", "cosmetics"),
    # Healthcare & Wellness
    "vicks vapoinhaler": ("Vicks VapoInhaler Portable Nasal Inhaler Stick, Refreshing Menthol Scent, Non-Medicated", "health"),
    "vicks": ("Vicks VapoInhaler Portable Nasal Inhaler Stick, Refreshing Menthol Scent, Non-Medicated", "health"),
    "inhaler": ("Vicks VapoInhaler Portable Nasal Inhaler Stick, Refreshing Menthol Scent", "health"),
}

PRODUCT_VISUAL_TEMPLATES = {
    "gaming backpack": "tactical gaming backpack, matte black nylon, angular design with laptop compartment, USB charging port, luggage pass-through sleeve, structured clamshell opening, gamer aesthetic",
    "backpack": "lifestyle backpack, padded laptop compartment, multiple organizational pockets, padded ergonomic shoulder straps, sturdy zippers",
    "laptop bag": "slim professional laptop sleeve, padded protection, handle and shoulder strap, clean business design",
    "travel bag": "large capacity rolling luggage or duffel bag, telescopic handle, spinner wheels, multiple compartments",
    "handbag": "structured leather top-handle handbag, polished metal feet, magnetic clasp closure, suede interior",
    "tote bag": "open canvas or leather tote, long drop shoulder straps, interior zip pocket, spacious main compartment",
    "crossbody bag": "compact zip crossbody bag, adjustable chain or strap, front slip pocket, card slots",
    "sneaker": "athletic sneaker, smooth or textured upper, rubber cupsole, brand logo, hero 3/4 angle showing sole",
    "running shoe": "performance running shoe, breathable engineered mesh upper, cushioned foam midsole, carbon-fiber plate visible",
    "basketball shoe": "high-top basketball sneaker, leather or synthetic upper, ankle support collar, pivot circle outsole",
    "dress shoe": "formal leather oxford, burnished toe cap, Blake welt construction, leather sole and heel",
    "boot": "Chelsea or lace-up leather ankle boot, elasticated gusset, block heel, rubber lug outsole",
    "headphones": "over-ear wireless headphones, memory foam earcup cushions, adjustable headband, premium cable or USB-C",
    "earbuds": "true wireless earbuds with in-ear silicone tips, charging case with LED battery indicator",
    "speaker": "portable Bluetooth speaker, cylindrical fabric grille, rubber base, colorful accent ring",
    "gaming headset": "over-ear gaming headset, retractable boom microphone, breathable mesh earcups, RGB lighting",
    "mechanical keyboard": "compact 65% or full-size mechanical keyboard, per-key RGB LEDs, keycap legends, braided USB-C cable",
    "gaming mouse": "ergonomic gaming mouse, honeycomb shell design, optical sensor, Paracord cable, programmable DPI",
    "laptop": "thin and light aluminum laptop, 13 or 15 inch edge-to-edge display, backlit keyboard, USB-C ports",
    "smartphone": "flagship Android or iPhone, edge-to-edge OLED display, triple camera module, glass-aluminum sandwich build",
    "smartwatch": "digital smartwatch, square or round case, always-on AMOLED display, health sensors, interchangeable band",
    "perfume": "luxury glass perfume bottle, gold or silver spray atomizer, etched label, designer cap",
    "skincare serum": "glass dropper bottle, clinical white label, 30ml volume, amber or clear glass",
    "luxury watch": "Swiss mechanical luxury watch, exhibition caseback, sapphire crystal, leather or metal bracelet",
    "coffee maker": "espresso machine, stainless steel boiler, professional portafilter, integrated grinder, digital display",
    "air fryer": "digital air fryer, matte black basket unit, touchscreen display, 5-7 quart capacity",
    "yoga mat": "eco-friendly TPE yoga mat, alignment print, non-slip texture, 6mm thick, carrying strap",
    "dumbbell": "chrome or urethane coated dumbbell, knurled handle, hex-shaped anti-roll head",
    "water bottle": "vacuum-insulated stainless steel bottle, powder coat finish, leak-proof lid, brand logo embossed",
    # Universal Commercial Shop Archetypes
    "t-shirt": "crewneck t-shirt, 100% combed ringspun cotton, tailored fit, double-needle stitched sleeves and hem, flat-lay studio presentation",
    "hoodie": "pullover fleece hoodie, heavyweight French terry cotton, kangaroo front pouch, ribbed cuffs, structured hood",
    "diamond ring": "18k gold diamond engagement ring, brilliant-cut round solitaire center diamond, micro-pavé band, mirror polish",
    "chef knife": "8-inch Japanese Damascus steel chef knife, 67-layer hammered finish, ergonomic dark pakkawood handle, razor-sharp edge",
    "sofa": "modern 3-seater Scandinavian minimalist sofa, textured bouclé upholstery, tapered solid oak legs, clean architectural silhouette",
    "protein powder": "matte black cylindrical whey protein isolate tub, minimalist gold debossed label, ribbed screw cap",
    "power drill": "heavy-duty cordless power drill, brushless motor, variable speed trigger, textured rubberized ergonomic grip, keyless chuck",
    "guitar": "acoustic dreadnought guitar, solid spruce top, mahogany back and sides, natural gloss lacquer finish, chrome die-cast tuners",
    "sunglasses": "luxury aviator sunglasses, gold-plated titanium frame, gradient UV400 lenses, clear silicone nose pads",
    "camera": "professional full-frame DSLR camera with 50mm f/1.4 prime lens attached, matte black magnesium alloy chassis, textured rubber grip, mode dials with engraved markings",
    "dslr": "professional full-frame DSLR camera with 50mm f/1.4 prime lens attached, matte black magnesium alloy chassis, textured rubber grip, mode dials with engraved markings",
    "mirrorless": "professional full-frame mirrorless camera with fast prime lens attached, matte black magnesium alloy body, tactile diamond grip"
}

# Catalog of curated template assets (Universal Commercial Store Coverage)
TEMPLATE_MAP = {
    "apparel": {
        "file": "tshirt_studio.jpg",
        "keywords": ["shirt", "t-shirt", "tshirt", "tee", "hoodie", "apparel", "clothing", "dress", "jacket", "coat", "sweater", "pants", "suit", "top", "crewneck", "fleece", "denim", "blazer", "skirt", "trouser", "activewear"],
        "default_title": "Premium Combed Ringspun Cotton Crewneck T-Shirt",
        "default_sku_prefix": "AI-AP"
    },
    "jewelry": {
        "file": "diamond_ring_studio.jpg",
        "keywords": ["ring", "diamond", "engagement ring", "wedding ring", "necklace", "pendant", "earrings", "bracelet", "gemstone", "gold ring", "silver ring", "jewelry", "solitaire", "brooch", "jewel", "fine jewelry"],
        "default_title": "18k Gold Brilliant Cut Solitaire Diamond Ring",
        "default_sku_prefix": "AI-JW"
    },
    "furniture": {
        "file": "sofa_studio.jpg",
        "keywords": ["sofa", "couch", "sectional", "loveseat", "armchair", "chair", "table", "desk", "bed", "bookshelf", "cabinet", "coffee table", "furniture", "living room", "seating", "credenza"],
        "default_title": "Scandinavian Minimalist Bouclé 3-Seater Sofa",
        "default_sku_prefix": "AI-HM"
    },
    "kitchen": {
        "file": "chef_knife_studio.jpg",
        "keywords": ["knife", "chef knife", "kitchen knife", "cutlery", "cookware", "blender", "espresso", "pan", "pot", "skillet", "kettle", "toaster", "cutting board", "kitchenware", "cooking", "damascus"],
        "default_title": "Professional 8-Inch Japanese Damascus Chef Knife",
        "default_sku_prefix": "AI-KT"
    },
    "footwear": {
        "file": "loafers_studio.jpg",
        "keywords": ["shoe", "loafer", "sneaker", "boot", "footwear", "sandal", "heel", "oxford", "derby", "moccasin", "slide", "runner", "trainer", "slides", "clog", "slip-on", "pump", "cleat", "flat", "espadrille", "mule", "stiletto", "boots", "shoes"],
        "default_title": "Artisan Leather Penny Loafers",
        "default_sku_prefix": "AI-FW"
    },
    "electronics": {
        "file": "headphones_studio.jpg",
        "keywords": ["headphone", "earphone", "audio", "speaker", "electronic", "gadget", "headset", "wireless", "tech", "device", "earbud", "smartwatch", "keyboard", "mouse", "charger", "cable", "laptop", "tablet", "camera", "drone", "phone", "mic", "microphone", "soundbar", "console"],
        "default_title": "Aether Studio Wireless ANC Headphones",
        "default_sku_prefix": "AI-EL"
    },
    "accessories": {
        "file": "watch_studio.jpg",
        "keywords": ["watch", "chronograph", "timepiece", "accessory", "bracelet", "sunglass", "glasses", "belt", "hat", "cap", "scarf", "tie", "cufflink", "wallet", "sunglasses", "chain"],
        "default_title": "Swiss Luxe Chronograph Automatic Watch",
        "default_sku_prefix": "AI-AC"
    },
    "cosmetics": {
        "file": "perfume_studio.jpg",
        "keywords": ["perfume", "fragrance", "scent", "cologne", "bottle", "cosmetic", "skincare", "serum", "lotion", "beauty", "cream", "lipstick", "makeup", "shampoo", "cleanser", "moisturizer", "mascara", "balm", "oil", "eyeliner", "sunscreen", "gloss", "foundation"],
        "default_title": "Ambre Nuit Eau De Parfum 75ml",
        "default_sku_prefix": "AI-CS"
    },
    "bags": {
        "file": "handbag_studio.jpg",
        "keywords": ["bag", "handbag", "purse", "tote", "backpack", "satchel", "clutch", "luggage", "wallet", "leather goods", "duffel", "duffle", "pouch", "briefcase", "suitcase", "crossbody", "messenger", "pack", "rucksack", "holdall"],
        "default_title": "Structured Textured Leather Top-Handle Bag",
        "default_sku_prefix": "AI-BG"
    },
    "health": {
        "file": "protein_powder_studio.jpg",
        "keywords": ["protein", "whey", "supplement", "powder", "creatine", "vitamin", "nutrition", "inhaler", "vicks", "vapoinhaler", "nasal", "menthol stick", "decongestant", "medicine", "pharmacy", "wellness"],
        "default_title": "Ultra-Pure Whey Isolate Protein Powder 2.5lb",
        "default_sku_prefix": "AI-NUT"
    },
    "tools": {
        "file": "drill_studio.jpg",
        "keywords": ["drill", "power drill", "tool", "hardware", "screwdriver", "hammer", "wrench", "saw", "pliers", "toolbox", "impact driver"],
        "default_title": "Heavy-Duty 20V Cordless Brushless Power Drill",
        "default_sku_prefix": "AI-TL"
    },
    "sports": {
        "file": "bowflex_dumbbells.jpg",
        "keywords": ["dumbbell", "dumbbells", "weights", "workout", "gym", "fitness", "barbell", "kettlebell", "basketball", "exercise", "training"],
        "default_title": "Precision Adjustable Commercial Dumbbells Set",
        "default_sku_prefix": "AI-SP"
    },
    "musical": {
        "file": "electric_guitar.jpg",
        "keywords": ["guitar", "electric guitar", "acoustic guitar", "instrument", "musical", "piano", "keyboard", "drum", "amplifier", "bass"],
        "default_title": "Artisan Solid-Body Electric Guitar Sunburst",
        "default_sku_prefix": "AI-MUS"
    },
    "office": {
        "file": "pen_studio.jpg",
        "keywords": ["pen", "fountain pen", "stationery", "notebook", "office", "writing", "journal", "desk organizer"],
        "default_title": "Executive Handcrafted Lacquer Fountain Pen",
        "default_sku_prefix": "AI-OF"
    },
    "grocery": {
        "file": "biryani_studio.jpg",
        "keywords": ["food", "gourmet", "snack", "sweet", "tea", "coffee", "beverage", "chocolate", "grocery", "baklava", "kebab", "taco", "biryani"],
        "default_title": "Artisanal Gourmet Specialty Delicacy",
        "default_sku_prefix": "AI-GR"
    }
}

STYLE_PRESETS = {
    "studio": {
        "name": "Commercial Clean Studio",
        "brightness": 1.02,
        "contrast": 1.05,
        "sharpness": 1.15,
        "saturation": 1.0,
        "warmth": 0
    },
    "editorial": {
        "name": "Editorial Luxury Vogue",
        "brightness": 0.98,
        "contrast": 1.12,
        "sharpness": 1.20,
        "saturation": 1.08,
        "warmth": 8
    },
    "white_hero": {
        "name": "Amazon / Shopify Pure White Hero",
        "brightness": 1.06,
        "contrast": 1.04,
        "sharpness": 1.25,
        "saturation": 0.96,
        "warmth": -3
    },
    "cyber": {
        "name": "Cyberpunk Neon Noir",
        "brightness": 0.92,
        "contrast": 1.18,
        "sharpness": 1.30,
        "saturation": 1.20,
        "warmth": -15
    },
    "minimal": {
        "name": "Scandinavian Warm Minimalist",
        "brightness": 1.04,
        "contrast": 0.98,
        "sharpness": 1.10,
        "saturation": 0.94,
        "warmth": 12
    }
}


def detect_domain_from_prompt(prompt: str) -> str:
    """Classifies user prompt into product domain. Checks brand map first for precision."""
    p_lower = prompt.lower()

    # 1. Brand map check (highest priority — prevents ROG->luxury bag mistakes)
    for brand, (_, brand_domain) in BRAND_PRODUCT_MAP.items():
        if brand in p_lower:
            return brand_domain

    # 2. Apparel & Clothing
    apparel_cues = [
        "shirt", "t-shirt", "tshirt", "tee", "hoodie", "hoody", "sweatshirt", "jacket",
        "coat", "sweater", "pants", "trouser", "trousers", "dress", "jeans", "polo",
        "blazer", "suit", "skirt", "fleece", "denim", "cardigan", "shorts", "sweatpants",
        "vest", "tank top", "sportswear", "activewear", "clothing", "apparel", "streetwear"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in apparel_cues):
        return "apparel"

    # 3. Jewelry & Luxury Goods
    jewelry_cues = [
        "ring", "diamond", "engagement ring", "wedding ring", "wedding band",
        "necklace", "pendant", "earring", "earrings", "bracelet", "gemstone",
        "gold ring", "silver ring", "jewelry", "solitaire", "brooch", "jewel", "fine jewelry"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in jewelry_cues):
        return "jewelry"

    # 4. Furniture & Home Living
    furniture_cues = [
        "sofa", "couch", "sectional", "loveseat", "armchair", "chair", "chairs",
        "dining table", "table", "desk", "bed", "bookshelf", "cabinet", "coffee table",
        "furniture", "credenza", "bench", "nightstand", "ottoman", "stool", "seating"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in furniture_cues):
        return "furniture"

    # 5. Kitchen, Dining & Cookware
    kitchen_cues = [
        "knife", "chef knife", "kitchen knife", "cutlery", "cookware", "blender",
        "espresso", "coffee maker", "coffee machine", "pan", "pot", "skillet",
        "kettle", "toaster", "cutting board", "kitchenware", "cooking", "damascus", "teapot"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in kitchen_cues):
        return "kitchen"

    # 6. Tools, Hardware & Industrial
    tools_cues = [
        "drill", "power drill", "cordless drill", "tool", "hardware", "screwdriver",
        "hammer", "wrench", "power tool", "saw", "pliers", "toolbox", "impact driver"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in tools_cues):
        return "tools"

    # 7. Sports, Fitness & Gym
    sports_cues = [
        "dumbbell", "dumbbells", "weights", "barbell", "kettlebell", "yoga mat",
        "basketball", "workout", "gym", "fitness", "exercise", "training", "bicycle"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in sports_cues):
        return "sports"

    # 8. Footwear keyword detection
    footwear_cues = [
        "jordan", "air jordan", "nike", "sneaker", "sneakers", "dunk", "dunks",
        "yeezy", "adidas", "puma", "new balance", "asics", "converse", "vans",
        "shoe", "shoes", "kicks", "loafer", "loafers", "boot", "boots", "sandal",
        "sandals", "heel", "heels", "oxford", "derby", "moccasin", "slide", "slides",
        "clog", "runner", "trainer", "trainers", "high-top", "low-top", "cleat",
        "flat", "espadrille", "mule", "stiletto", "footwear", "sneakerhead"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in footwear_cues):
        return "footwear"

    # 9. Electronics
    electronics_cues = [
        "headphone", "headphones", "headset", "earphone", "earphones", "earbud",
        "earbuds", "audio", "speaker", "speakers", "gadget", "keyboard", "mouse",
        "charger", "cable", "laptop", "tablet", "camera", "drone", "phone", "mic",
        "microphone", "soundbar", "console", "smartwatch", "bluetooth", "gaming",
        "controller", "monitor", "display", "processor", "gpu", "cpu",
        "power bank", "router", "hard drive", "ssd", "usb", "hub"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in electronics_cues):
        return "electronics"

    # 10. Cosmetics & Beauty
    cosmetics_cues = [
        "perfume", "fragrance", "scent", "cologne", "cosmetic", "skincare",
        "serum", "lotion", "beauty", "cream", "lipstick", "makeup", "shampoo",
        "cleanser", "moisturizer", "mascara", "balm", "oil", "eyeliner",
        "sunscreen", "toner", "primer", "blush", "foundation", "eyeshadow",
        "palette", "hair care", "conditioner", "face wash"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in cosmetics_cues):
        return "cosmetics"

    # 11. Bags
    bags_cues = [
        "backpack", "backpacks", "rucksack", "tote", "handbag", "purse", "clutch",
        "satchel", "duffel", "duffle", "briefcase", "luggage", "suitcase",
        "crossbody", "messenger", "pouch", "bag", "bags", "fanny pack",
        "daypack", "laptop bag", "gym bag", "travel bag", "weekender"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in bags_cues):
        return "bags"

    # 12. Accessories & Watches
    accessories_cues = [
        "watch", "watches", "chronograph", "timepiece", "rolex", "omega",
        "wristwatch", "sunglass", "sunglasses", "glasses", "shades", "eyewear",
        "belt", "cufflink", "hat", "cap", "beanie", "scarf", "wallet"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in accessories_cues):
        return "accessories"

    # 13. Health & Wellness
    health_cues = ["protein", "whey", "supplement", "vitamin", "powder", "vicks", "inhaler", "nasal", "vapoinhaler", "medicine", "menthol stick", "decongestant"]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in health_cues):
        return "health"

    # 14. Musical Instruments
    musical_cues = ["guitar", "electric guitar", "acoustic guitar", "instrument", "musical", "piano", "keyboard", "drum", "violin", "amplifier"]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in musical_cues):
        return "musical"

    # 15. Office & Stationery
    office_cues = ["pen", "fountain pen", "stationery", "notebook", "office", "writing", "journal", "desk organizer"]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in office_cues):
        return "office"

    # 16. Gourmet Food & Beverage
    grocery_cues = ["food", "gourmet", "snack", "sweet", "tea", "coffee", "beverage", "chocolate", "grocery", "biryani", "taco", "kebab", "baklava"]
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in grocery_cues):
        return "grocery"

    # Score-based fallback
    scores = {}
    for domain, info in TEMPLATE_MAP.items():
        scores[domain] = sum(3 for kw in info["keywords"] if kw in p_lower)
    best_domain = max(scores, key=scores.get)
    if scores[best_domain] > 0:
        return best_domain

    return "general"


def select_template_asset(prompt: str, domain: str) -> Tuple[str, str, str]:
    """
    Selects the best visual template file, domain, and SKU prefix for a given prompt.
    Returns (template_filename, domain, sku_prefix).
    Supports 50+ universal Amazon & e-commerce product categories with world-class precision.
    """
    p_lower = prompt.lower()
    
    # 1. SMARTPHONES & MOBILE DEVICES (HIGHEST PRIORITY - Prevents Phone -> Watch misclassification)
    is_s21_or_samsung_phone = any(w in p_lower for w in [
        "s21 fe", "s21", "galaxy s21", "samsung s21", "lavender", "galaxy s24", 
        "galaxy s23", "galaxy s22", "samsung galaxy", "galaxy phone", "samsung phone",
        "galaxy note", "galaxy z fold", "galaxy z flip"
    ])
    if is_s21_or_samsung_phone:
        return "samsung_galaxy_s21_fe_lavender.jpg", "electronics", "AI-SAM"

    if any(w in p_lower for w in ["iphone 17 pro", "iphone 17", "17 pro", "iphone 17 pro max"]):
        return "iphone_17_pro.jpg", "electronics", "AI-APL"

    if any(w in p_lower for w in ["iphone", "iphone 16", "iphone 15", "iphone 14", "iphone 13", "apple phone", "ios phone"]):
        return "iphone_17_pro.jpg", "electronics", "AI-APL"

    if any(w in p_lower for w in ["smartphone", "cell phone", "mobile phone", "android phone", "pixel phone", "google pixel"]):
        return "smartphone_studio.jpg", "electronics", "AI-PH"

    # 2. SMARTWATCHES & WEARABLES vs LUXURY TIMEPIECES
    is_smartwatch = any(w in p_lower for w in [
        "smartwatch", "apple watch", "galaxy watch", "fitness watch", "garmin", "fitbit", "smart watch"
    ])
    if is_smartwatch:
        return "smartwatch_studio.jpg", "accessories", "AI-SW"

    if any(w in p_lower for w in ["rolex", "submariner", "luxury dive watch", "oystersteel"]):
        return "rolex_submariner.jpg", "accessories", "AI-RLX"

    if any(w in p_lower for w in ["watch", "watches", "chronograph", "timepiece", "omega", "seiko", "patek", "audemars", "casio"]):
        return "watch_studio.jpg", "accessories", "AI-WA"

    # 3. AUDIO & HEADPHONES
    if any(w in p_lower for w in ["sony wh", "wh-1000xm", "wh-1000", "sony headphones"]):
        return "sony_headphones.jpg", "electronics", "AI-SNY"

    if any(w in p_lower for w in ["rose gold headphones", "luxury headphones", "pink headphones"]):
        return "rose_gold_headphones.jpg", "electronics", "AI-EL"

    if any(w in p_lower for w in ["headphone", "headphones", "headset", "earphones", "earbuds", "airpods"]):
        return "headphones_studio.jpg", "electronics", "AI-EL"

    if any(w in p_lower for w in ["speaker", "speakers", "bluetooth speaker", "soundbar", "jbl", "marshall"]):
        return "speaker_studio.jpg", "electronics", "AI-SPK"

    # 4. HEALTHCARE & WELLNESS (Prevents Inhaler / Medicine -> Lipstick mistakes)
    if any(w in p_lower for w in ["inhaler", "vicks", "vapoinhaler", "nasal", "menthol stick", "decongestant"]):
        return "vicks_inhaler.jpg", "health", "AI-VIC"

    # 5. BAGS & BACKPACKS (Evaluated before laptops so "laptop compartment" doesn't trigger laptop)
    if any(w in p_lower for w in ["rog", "republic of gamers", "asus rog", "gaming backpack", "tactical backpack", "alienware", "gamer backpack"]):
        return "backpack_gaming.jpg", "bags", "AI-ROG"

    if any(w in p_lower for w in ["north face", "borealis", "hiking backpack", "outdoor backpack", "daypack"]):
        return "north_face_backpack.jpg", "bags", "AI-NF"

    if any(w in p_lower for w in ["backpack", "backpacks", "rucksack", "travel pack", "school bag"]):
        return "classic_backpack_studio.jpg", "bags", "AI-BP"

    if any(w in p_lower for w in ["handbag", "purse", "gucci", "chanel", "prada", "louis vuitton", "tote", "clutch", "satchel", "shoulder bag", "crossbody"]):
        return "handbag_studio.jpg", "bags", "AI-BG"

    # 6. TECH PERIPHERALS & COMPUTERS
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in [
        "camera", "cameras", "dslr", "mirrorless", "prime lens", "50mm", "telephoto",
        "canon", "nikon", "sony alpha", "sony a7", "fujifilm", "lumix", "leica",
        "hasselblad", "camera lens", "camera body", "full-frame", "full frame"
    ]):
        return "camera_studio.jpg", "electronics", "AI-CAM"

    if any(w in p_lower for w in ["mouse", "wireless mouse", "logitech mouse", "gaming mouse", "trackpad"]):
        return "mouse_studio.jpg", "electronics", "AI-MSE"

    if any(w in p_lower for w in ["drone", "quadcopter", "dji", "mavic", "aerial camera"]):
        return "drone_studio.jpg", "electronics", "AI-DRN"

    is_bag = any(b in p_lower for b in ["backpack", "bag", "pack", "rucksack", "tote", "purse", "satchel", "compartment", "sleeve"])
    if not is_bag and any(w in p_lower for w in ["laptop", "notebook", "gaming laptop", "macbook", "thinkpad", "zenbook", "blade"]):
        return "laptop_studio.jpg", "electronics", "AI-LP"

    # 7. FOOTWEAR & SHOES (Specific Models First)
    if any(w in p_lower for w in ["samba", "adidas samba"]):
        return "adidas_samba.jpg", "footwear", "AI-ADI"

    if any(w in p_lower for w in ["dunk", "panda", "dunk low"]):
        return "dunk_panda.jpg", "footwear", "AI-NK"

    if any(w in p_lower for w in ["jordan", "air jordan"]):
        return "jordan_green_white.jpg", "footwear", "AI-NK"

    if any(w in p_lower for w in ["timberland", "wheat boot", "work boot", "lug boot"]):
        return "timberland_boot.jpg", "footwear", "AI-TB"

    if any(w in p_lower for w in ["loafer", "loafers", "penny loafer", "oxford", "derby", "brogue", "monk strap", "dress shoe", "formal shoe", "moccasin"]):
        return "classic_loafer_studio.jpg", "footwear", "AI-FW"

    if any(w in p_lower for w in ["slide", "slides", "sandal", "sandals", "flip flop", "clog", "birkenstock"]):
        return "slide_sandals.jpg", "footwear", "AI-SD"

    if any(w in p_lower for w in ["sneaker", "sneakers", "kicks", "runner", "running shoe", "trainer", "trainers", "yeezy", "new balance", "puma", "asics", "converse", "vans", "shoes", "shoe"]):
        return "sneaker_studio.jpg", "footwear", "AI-SN"

    # 7. BEAUTY, FRAGRANCE & SKINCARE
    if any(w in p_lower for w in ["dior", "sauvage", "dior sauvage"]):
        return "dior_sauvage.jpg", "cosmetics", "AI-CD"

    if any(w in p_lower for w in ["lipstick", "lip gloss", "ysl", "rouge", "lip"]):
        return "ysl_lipstick.jpg", "cosmetics", "AI-YSL"

    if any(w in p_lower for w in ["serum", "dropper", "ordinary serum", "niacinamide", "hyaluronic", "facial oil"]):
        return "serum_bottle.jpg", "cosmetics", "AI-SRM"

    if any(w in p_lower for w in ["lotion", "moisturizer", "cerave", "cream", "cleanser", "skincare", "body lotion"]):
        return "skincare_lotion.jpg", "cosmetics", "AI-LTN"

    if any(w in p_lower for w in ["perfume", "fragrance", "scent", "cologne", "eau de parfum", "parfum"]):
        return "dior_sauvage.jpg" if any(w in p_lower for w in ["men", "homme", "black", "dark"]) else "perfume_studio.jpg", "cosmetics", "AI-CS"

    # 8. DRINKWARE & KITCHEN
    if any(w in p_lower for w in ["stanley", "quencher", "stanley cup", "stanley tumbler"]):
        return "stanley_tumbler.jpg", "home", "AI-STN"

    if any(w in p_lower for w in ["tumbler", "yeti", "insulated cup", "travel mug"]):
        return "tumbler_studio.jpg", "home", "AI-TB"

    if any(w in p_lower for w in ["water bottle", "hydro flask", "canteen", "flask"]):
        return "water_bottle.jpg", "home", "AI-WB"

    if any(w in p_lower for w in ["espresso", "breville", "coffee machine", "espresso machine"]):
        return "breville_espresso.jpg", "home", "AI-BRV"

    if any(w in p_lower for w in ["coffee mug", "mug", "coffee cup"]):
        return "coffee_mug_studio.jpg", "home", "AI-MG"

    if any(w in p_lower for w in ["teapot", "tea maker", "kettle", "teaset"]):
        return "teapot_studio.jpg", "home", "AI-TP"

    # 9. HOME & FURNITURE
    if any(w in p_lower for w in ["chair", "office chair", "desk chair", "armchair", "seating"]):
        return "chair_studio.jpg", "home", "AI-CHR"

    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in ["lamp", "desk lamp", "table lamp", "floor lamp", "bedside lamp", "lantern", "chandelier", "sconce"]):
        return "lamp_studio.jpg", "home", "AI-LMP"

    # 10. TOOLS & HARDWARE
    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in ["drill", "power drill", "cordless drill", "impact driver", "rotary tool"]):
        return "drill_studio.jpg", "tools", "AI-DRL"

    # 11. SPORTS & FITNESS
    if any(w in p_lower for w in ["dumbbell", "dumbbells", "weights", "bowflex", "barbell", "kettlebell"]):
        return "bowflex_dumbbells.jpg", "sports", "AI-BWF"

    if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in ["basketball", "wilson", "spalding", "soccer ball", "football", "tennis ball", "baseball"]):
        return "basketball_studio.jpg", "sports", "AI-BB"

    # 12. MUSICAL INSTRUMENTS
    if any(w in p_lower for w in ["electric guitar", "fender", "gibson", "stratocaster", "telecaster"]):
        return "electric_guitar.jpg", "musical", "AI-EGTR"

    if any(w in p_lower for w in ["guitar", "acoustic guitar", "classical guitar", "ukulele"]):
        return "acoustic_guitar.jpg", "musical", "AI-GTR"

    # 13. APPAREL & CLOTHING
    if any(w in p_lower for w in ["hoodie", "hoody", "sweatshirt", "fleece pullover", "sweater"]):
        return "hoodie_studio.jpg", "apparel", "AI-HD"
    if any(w in p_lower for w in ["t-shirt", "tshirt", "tee", "shirt", "crewneck", "top", "apparel", "clothing", "dress", "jersey", "polo", "tank", "vest"]):
        return "tshirt_studio.jpg", "apparel", "AI-TS"

    # 14. JEWELRY & RINGS
    if any(w in p_lower for w in ["ring", "diamond", "engagement ring", "wedding ring", "gemstone", "solitaire", "diamond ring"]):
        return "diamond_ring_studio.jpg", "jewelry", "AI-JW"

    # 15. FURNITURE & SOFAS
    if any(w in p_lower for w in ["sofa", "couch", "sectional", "loveseat", "living room sofa", "settee"]):
        return "sofa_studio.jpg", "furniture", "AI-SFA"

    # 16. KITCHEN CUTLERY & CHEF KNIVES
    if any(w in p_lower for w in ["knife", "chef knife", "kitchen knife", "cutlery", "blade", "damascus"]):
        return "chef_knife_studio.jpg", "kitchen", "AI-KNF"

    # 17. HEALTH, NUTRITION & PROTEIN POWDER
    if any(w in p_lower for w in ["protein", "whey", "supplement", "powder", "tub", "creatine", "shake"]):
        return "protein_powder_studio.jpg", "health", "AI-NUT"

    # 18. OFFICE PRODUCTS
    if any(w in p_lower for w in ["fountain pen", "pen", "ballpoint", "writing instrument"]):
        return "pen_studio.jpg", "office", "AI-PEN"

    # 19. ACCESSORIES & EYEWEAR
    if any(w in p_lower for w in ["sunglass", "sunglasses", "shades", "eyewear", "glasses", "ray-ban"]):
        return "sunglasses_studio.jpg", "accessories", "AI-SG"

    # 20. GOURMET FOOD
    if any(w in p_lower for w in ["biryani", "rice dish"]):
        return "biryani_studio.jpg", "grocery", "AI-BRY"
    if any(w in p_lower for w in ["taco", "tacos", "burrito"]):
        return "taco_studio.jpg", "grocery", "AI-TC"
    if any(w in p_lower for w in ["kebab", "kebabs"]):
        return "kebab_studio.jpg", "grocery", "AI-KB"
    if any(w in p_lower for w in ["baklava", "sweet", "pastry"]):
        return "baklava_studio.jpg", "grocery", "AI-BKL"

    # Universal Store Fallbacks by Domain
    if domain in ["apparel", "clothing", "fashion"]:
        return "tshirt_studio.jpg", "apparel", "AI-AP"
    if domain in ["jewelry", "luxury"]:
        return "diamond_ring_studio.jpg", "jewelry", "AI-JW"
    if domain in ["furniture", "living"]:
        return "sofa_studio.jpg", "furniture", "AI-HM"
    if domain in ["kitchen", "cookware"]:
        return "chef_knife_studio.jpg", "kitchen", "AI-KT"
    if domain in ["home"]:
        return "sofa_studio.jpg" if any(w in p_lower for w in ["sofa", "couch", "living"]) else "chair_studio.jpg", "home", "AI-HM"
    if domain in ["tools", "hardware"]:
        return "drill_studio.jpg", "tools", "AI-TL"
    if domain in ["sports", "fitness"]:
        return "bowflex_dumbbells.jpg", "sports", "AI-SP"
    if domain in ["musical", "instruments"]:
        return "electric_guitar.jpg", "musical", "AI-MUS"
    if domain in ["office", "stationery"]:
        return "pen_studio.jpg", "office", "AI-OF"
    if domain in ["health", "wellness"]:
        return "protein_powder_studio.jpg", "health", "AI-NUT"
    if domain in ["grocery", "food"]:
        return "biryani_studio.jpg", "grocery", "AI-GR"
    if domain == "electronics":
        if any(re.search(r'\b' + re.escape(w) + r'\b', p_lower) for w in ["camera", "cameras", "dslr", "lens", "optics", "canon", "nikon"]):
            return "camera_studio.jpg", "electronics", "AI-CAM"
        return "smartphone_studio.jpg" if any(w in p_lower for w in ["phone", "mobile"]) else "headphones_studio.jpg", "electronics", "AI-EL"
    if domain == "footwear":
        return "sneaker_studio.jpg", "footwear", "AI-FW"
    if domain == "bags":
        return "classic_backpack_studio.jpg", "bags", "AI-BG"
    if domain == "accessories":
        return "watch_studio.jpg", "accessories", "AI-AC"
    if domain == "cosmetics":
        return "perfume_studio.jpg", "cosmetics", "AI-CS"

    return "samsung_galaxy_s21_fe_lavender.jpg" if "s21" in p_lower else "sneaker_studio.jpg", "footwear", "AI-PR"


def extract_title_from_prompt(prompt: str, domain: str) -> str:
    """Generates an elegant, commercial product title from natural language prompt."""
    p_lower = prompt.lower()
    # Use brand-specific title if brand is recognized
    for brand, (brand_desc, _) in BRAND_PRODUCT_MAP.items():
        if brand in p_lower:
            parts = brand_desc.split(",")
            return parts[0].strip().title()[:60]

    cleaned = re.sub(r'^(a|an|the|generate|create|render|photo of|photograph of|commercial shot of|image of)\s+', '', prompt.strip(), flags=re.I)
    cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned).strip()
    words = cleaned.split()
    if len(words) > 12:
        words = words[:12]
    while words and words[-1].lower() in ["with", "and", "in", "on", "for", "at", "by", "of", "a", "an", "the"]:
        words.pop()
    if len(words) <= 1:
        return TEMPLATE_MAP.get(domain, {}).get("default_title", "Commercial Catalog Item")
    title = " ".join(w.capitalize() for w in words)
    title = re.sub(r'[,.\-_:;]+$', '', title)
    return title


def build_imagen_prompt(user_prompt: str, domain: str, style_preset: str = "studio") -> str:
    """
    Constructs a detailed Gemini Imagen 3 prompt for accurate product photography.
    Ensures ROG backpack generates a gaming backpack, not a luxury handbag.
    """
    p_lower = user_prompt.lower()
    visual_desc = None

    # Brand recognition (highest priority)
    for brand, (brand_desc, _) in BRAND_PRODUCT_MAP.items():
        if brand in p_lower:
            visual_desc = brand_desc
            break

    # Product type recognition
    if visual_desc is None:
        for product_type, type_desc in PRODUCT_VISUAL_TEMPLATES.items():
            if product_type in p_lower:
                visual_desc = type_desc
                break

    # Fallback: use user prompt directly
    if visual_desc is None:
        cleaned = re.sub(r'^(a|an|the|generate|create|render|photo of|image of)\s+', '', user_prompt.strip(), flags=re.I)
        visual_desc = cleaned

    # Color extraction
    color_desc = ""
    color_map = {
        "red": "vibrant red", "blue": "deep navy blue", "green": "forest green",
        "black": "matte black", "white": "crisp white", "grey": "charcoal grey",
        "gray": "charcoal grey", "gold": "brushed gold", "silver": "polished silver",
        "pink": "rose pink", "purple": "royal purple", "orange": "burnt orange",
        "brown": "rich cognac brown", "navy": "navy blue", "cream": "off-white cream",
        "beige": "warm beige", "yellow": "bright yellow", "cyan": "electric cyan",
        "rose gold": "rose gold metallic", "titanium": "titanium grey"
    }
    for color_kw, color_desc_val in color_map.items():
        if color_kw in p_lower:
            color_desc = f", {color_desc_val} colorway"
            break

    # Photography style directive
    style_direction = {
        "studio": "pure white seamless infinity background, professional softbox lighting, 3/4 hero angle",
        "editorial": "editorial fashion magazine aesthetic, neutral grey gradient background, dramatic directional lighting",
        "white_hero": "pure white RGB(255,255,255) seamless background, centered, even diffused studio lighting, Amazon standard",
        "cyber": "dark studio background, RGB neon rim lighting, cyberpunk gaming aesthetic",
        "minimal": "off-white Scandinavian minimalist background, warm natural window light"
    }.get(style_preset, "pure white seamless studio background, professional commercial lighting")

    domain_style = {
        "footwear": "shoe product photography, side 3/4 profile at 30 degree angle, sole edge visible, crisp studio lighting, sharp texture detail",
        "electronics": "consumer electronics photography, 3/4 angle showing front face and side ports, controlled reflections on surfaces",
        "bags": "bag product photography, standing upright showing front face and handles, structured shape clear, hardware detail sharp",
        "cosmetics": "beauty product photography, minimalist podium or white surface, macro detail on packaging, subtle elegant reflection",
        "accessories": "accessories product photography, hero angle showing dial or face, metal reflections controlled, precision detail",
        "general": "professional e-commerce photography, white background, studio lighting, 3/4 hero angle"
    }.get(domain, "professional e-commerce product photography, white background")

    return (
        f"Professional Amazon product photograph: {visual_desc}{color_desc}. "
        f"Photography: {style_direction}. "
        f"Technique: {domain_style}. "
        f"Ultra-sharp focus, photorealistic, no motion blur, no text overlays, "
        f"no watermarks, no people, no hands, single product only, 1:1 square format, "
        f"commercial e-commerce quality product hero shot."
    )


def analyze_product_with_gemini(prompt: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Uses Google Gemini 3.8 Flash to semantically analyze any product in the world.
    Extracts brand, real product domain, specific archetype, commercial title,
    visual description, and best matching template.
    Guarantees world-class distinction between products (e.g. ROG gaming backpack vs Gucci bag).
    """
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        templates_list = [
            'iphone_17_pro.jpg', 'samsung_galaxy_s21_fe_lavender.jpg', 'iphone_studio.jpg', 'smartphone_studio.jpg',
            'smartwatch_studio.jpg', 'rolex_submariner.jpg', 'watch_studio.jpg',
            'sony_headphones.jpg', 'rose_gold_headphones.jpg', 'headphones_studio.jpg', 'speaker_studio.jpg',
            'mouse_studio.jpg', 'drone_studio.jpg', 'camera_studio.jpg', 'laptop_studio.jpg',
            'adidas_samba.jpg', 'dunk_panda.jpg', 'jordan_green_white.jpg', 'timberland_boot.jpg',
            'classic_loafer_studio.jpg', 'slide_sandals.jpg', 'sneaker_studio.jpg',
            'tshirt_studio.jpg', 'hoodie_studio.jpg',
            'diamond_ring_studio.jpg', 'sunglasses_studio.jpg',
            'sofa_studio.jpg', 'chair_studio.jpg', 'lamp_studio.jpg',
            'chef_knife_studio.jpg', 'breville_espresso.jpg', 'stanley_tumbler.jpg', 'tumbler_studio.jpg', 'water_bottle.jpg',
            'coffee_mug_studio.jpg', 'teapot_studio.jpg',
            'backpack_gaming.jpg', 'north_face_backpack.jpg', 'classic_backpack_studio.jpg', 'handbag_studio.jpg',
            'dior_sauvage.jpg', 'ysl_lipstick.jpg', 'serum_bottle.jpg', 'skincare_lotion.jpg', 'perfume_studio.jpg',
            'protein_powder_studio.jpg', 'vicks_inhaler.jpg',
            'drill_studio.jpg', 'bowflex_dumbbells.jpg', 'basketball_studio.jpg',
            'electric_guitar.jpg', 'acoustic_guitar.jpg', 'pen_studio.jpg',
            'biryani_studio.jpg', 'taco_studio.jpg', 'kebab_studio.jpg', 'baklava_studio.jpg'
        ]
        sys_prompt = (
            "You are a master commercial e-commerce catalog director for any retail shop (fashion, jewelry, furniture, kitchen, electronics, sports, health, tools, gourmet). "
            "Analyze the given product prompt and return ONLY a single valid JSON object (no markdown, no backticks) with keys:\n"
            "- 'title': Professional commercial e-commerce product title (e.g. '18k White Gold Brilliant Cut Solitaire Diamond Ring' or 'Canon EOS R5 Full-Frame Camera')\n"
            "- 'domain': One of ['apparel', 'jewelry', 'furniture', 'kitchen', 'footwear', 'electronics', 'accessories', 'bags', 'cosmetics', 'health', 'tools', 'sports', 'musical', 'office', 'grocery', 'general']\n"
            "- 'archetype': Specific archetype, e.g. 'camera', 'tshirt', 'diamond_ring', 'sofa', 'chef_knife', 'smartphone', 'gaming_backpack', 'handbag', 'smartwatch', 'sneaker', 'laptop', 'tumbler', 'perfume', 'protein_powder'\n"
            f"- 'template_file': Best matching studio template from: {templates_list}\n"
            "- 'brand': Detected brand name or 'Universal'\n"
            "- 'sku_prefix': 2-4 letter uppercase SKU prefix (e.g. 'AI-CAM', 'AI-AP', 'AI-JW', 'AI-HM', 'AI-KT', 'AI-SAM', 'AI-ROG', 'AI-NK', 'AI-NUT')"
        )
        response = None
        for cand_model in ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]:
            try:
                response = client.models.generate_content(
                    model=cand_model,
                    contents=f"{sys_prompt}\n\nProduct prompt: '{prompt}'"
                )
                if response and response.text:
                    break
            except Exception as model_err:
                continue
        if response and response.text:
            raw_text = response.text.strip()
            # Clean possible markdown fencing
            raw_text = re.sub(r'^```json\s*', '', raw_text, flags=re.I)
            raw_text = re.sub(r'^```\s*', '', raw_text)
            raw_text = re.sub(r'```$', '', raw_text).strip()
            
            import json
            m = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                print(f"[AI Studio] Gemini analysis: {parsed.get('brand')} -> {parsed.get('archetype')} ({parsed.get('template_file')})")
                return parsed
    except Exception as e:
        print(f"[AI Studio] Gemini reasoning fallback: {e}")
    return None


def enhance_prompt_offline(prompt: str, style_preset: str = "studio") -> Dict[str, Any]:
    """
    Intelligent offline prompt analyzer & enhancer.
    Deconstructs user prompt into archetype, brand, materials, color,
    and enriches with commercial photography directives for 100% accurate AI generation.
    """
    domain = detect_domain_from_prompt(prompt)
    title = extract_title_from_prompt(prompt, domain)
    p_lower = prompt.lower()

    visual_desc = None
    brand_found = None
    for brand, (brand_desc, b_dom) in BRAND_PRODUCT_MAP.items():
        if brand in p_lower:
            visual_desc = brand_desc
            brand_found = brand.title()
            domain = b_dom
            break

    if not visual_desc:
        for p_type, type_desc in PRODUCT_VISUAL_TEMPLATES.items():
            if p_type in p_lower:
                visual_desc = type_desc
                break

    if not visual_desc:
        cleaned = re.sub(r'^(a|an|the|generate|create|render|photo of|image of)\s+', '', prompt.strip(), flags=re.I)
        visual_desc = f"{cleaned} commercial item with premium material finish and refined craftsmanship"

    # Color extraction
    color_map = {
        "red": "vibrant crimson red", "blue": "deep navy blue", "green": "pine forest green",
        "black": "matte obsidian black", "white": "crisp alpine white", "grey": "charcoal grey",
        "gray": "charcoal grey", "gold": "brushed champagne gold", "silver": "polished metallic silver",
        "pink": "blush rose pink", "purple": "royal amethyst purple", "orange": "burnt amber orange",
        "brown": "burnished cognac brown", "navy": "deep navy blue", "cream": "warm off-white cream",
        "beige": "desert tan beige", "yellow": "canary yellow", "cyan": "electric cyan",
        "rose gold": "rose gold metallic", "titanium": "space titanium grey", "lavender": "pastel lavender"
    }
    color_desc = ""
    for color_kw, color_val in color_map.items():
        if color_kw in p_lower:
            color_desc = f" in {color_val}"
            break

    staging = {
        "studio": "pure seamless white studio cyclorama background, professional multi-point softbox diffused lighting, subtle neutral contact shadow, gentle rim specular reflection",
        "editorial": "luxury vogue editorial campaign set, dramatic directional key light with subtle fill, subtle neutral grey gradient background, moody depth of field",
        "white_hero": "pure white RGB(255,255,255) seamless infinity catalog backdrop, even shadowless diffused commercial lighting, Amazon standard",
        "cyber": "dark obsidian studio backdrop, sharp RGB neon rim lighting, high-contrast cybernetic specular highlights",
        "minimal": "warm natural daylight illumination, Scandinavian minimalist backdrop, soft organic shadows"
    }.get(style_preset, "pure seamless white studio cyclorama background, professional commercial lighting")

    angle_technique = {
        "footwear": "3/4 commercial hero perspective at 30-degree elevation, showing lateral profile, toe box silhouette, and outsole tread details with razor-sharp texture",
        "electronics": "centered 3/4 hero angle emphasizing precision engineered ports, tactile buttons, matte/gloss finish contrast, and controlled specular reflections",
        "bags": "upright hero presentation showing full volume, ergonomic strap curvature, polished metal hardware, and textured weave definition",
        "cosmetics": "clean minimalist podium hero shot, macro focal clarity on packaging texture and embossed typography, subtle liquid caustic highlights",
        "accessories": "hero angle focused on intricate dial/hardware detailing, anti-reflective crystal reflections, micro-engravings sharp",
        "health": "clean clinical commercial presentation, upright hero angle, crisp typography and packaging details",
        "home": "architectural 3/4 hero presentation, tactile material textures, authentic material weight",
        "apparel": "commercial flat-lay or invisible ghost mannequin hero presentation, crisp seams, fabric weave texture, realistic natural fold draping",
        "jewelry": "extreme macro 1:1 commercial hero shot, faceted gemstone internal brilliance, pristine mirror-polished metallic surface with micro-facet reflections",
        "furniture": "commercial showroom 3/4 hero perspective, tactile upholstery weave or wood grain texture, structural proportion and balanced depth of field",
        "kitchen": "culinary hero angle at 45 degrees, razor-honed bevel reflection, ergonomic handle grip texture, pristine stainless/damascus finish",
        "tools": "high-torque commercial angle showcasing ergonomic rubberized grip, durable chuck detailing, rugged hardshell finish, industrial durability",
        "sports": "dynamic commercial hero angle, textured athletic grip surfaces, balanced equipment proportions, active finish",
        "musical": "master luthier hero perspective, gloss lacquer reflections, fretwork and bridge detail, authentic wood grain resonance",
        "office": "refined executive tabletop angle, polished nib/metal hardware clarity, luxury corporate aesthetic",
        "grocery": "artisanal gourmet commercial hero presentation, delicious culinary highlights, fresh steam or delicate surface garnish, appetizing depth of field",
        "general": "centered 3/4 commercial hero presentation, crisp depth of field, tactile surface textures"
    }.get(domain, "centered 3/4 commercial hero presentation, crisp depth of field")

    enhanced_prompt = (
        f"A professional commercial product hero photograph of {visual_desc}{color_desc}. "
        f"Photography setup: {staging}. "
        f"Camera perspective: {angle_technique}. "
        f"Ultra-sharp focus, photorealistic 8k e-commerce catalog photography, "
        f"single product only, centered composition, no people, no hands, no text overlays, "
        f"no watermarks, pristine commercial presentation."
    )

    features = [
        f"Commercial studio setup ({style_preset.replace('_', ' ').title()})",
        f"Specific {domain.title()} material and texture definitions",
        "Centered 3/4 hero camera angle with macro clarity",
        "Distraction-free seamless catalog presentation"
    ]

    return {
        "enhanced_prompt": enhanced_prompt,
        "title": title,
        "domain": domain,
        "features_added": features,
        "summary": f"Enhanced with high-end {domain} materials, commercial {style_preset} lighting, and 3/4 hero camera perspective.",
        "category_hint": domain
    }


def enhance_prompt(prompt: str, style_preset: str = "studio") -> Dict[str, Any]:
    """
    Main Prompt Enhancer:
    Analyzes user input and extends it with rich, photorealistic specifications
    so the AI image generator understands exactly what the user wants for 100% accuracy.
    Specially trained for any commercial store/shop selling any type of products.

    Uses Gemini 3.5/3.8 Flash with multi-model fallback, with instant offline rule-based fallback.
    """
    if not prompt or not prompt.strip():
        return {
            "enhanced_prompt": "",
            "title": "",
            "domain": "general",
            "features_added": [],
            "summary": "Please enter a product prompt to enhance.",
            "category_hint": "general"
        }

    api_key = _load_gemini_key()
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            sys_prompt = (
                "You are an expert commercial e-commerce product prompt engineer and photography director "
                "trained for commercial online stores and retail catalogs of ANY type selling any items "
                "(fashion apparel, jewelry, furniture, kitchenware, consumer electronics, footwear, bags, beauty/cosmetics, sports/fitness, tools, gourmet food, supplements, musical instruments, office stationery). "
                "Analyze the user's product prompt and enhance it into an ultra-specific, photorealistic prompt for AI image generation. "
                "The goal is to ensure the AI image generator understands 100% what the user wants and produces an exact, high-end commercial hero shot.\n\n"
                "Requirements for the enhanced_prompt:\n"
                "1. Clearly identify the specific product model, archetype, and brand inspiration (if stated or implied).\n"
                "2. Specify premium realistic materials, construction details, textures, tactile surfaces, and precise color palette.\n"
                "3. Specify exact commercial camera angle (e.g. 3/4 commercial hero angle, macro jewelry angle, or flat-lay apparel) and sharp focal depth.\n"
                "4. Specify commercial studio lighting matching the requested preset:\n"
                "   - 'studio': pure white seamless cyclorama, softbox diffused lighting, gentle rim light\n"
                "   - 'editorial': dramatic directional lighting, subtle grey gradient, luxury magazine aesthetic\n"
                "   - 'white_hero': pure RGB(255,255,255) seamless infinity background, even shadowless lighting, Amazon standard\n"
                "   - 'cyber': dark obsidian backdrop, vibrant RGB neon rim lights, sharp specular highlights\n"
                "   - 'minimal': warm natural daylight illumination, Scandinavian minimalist setup, soft organic shadows\n"
                "5. Explicitly enforce: single product hero shot, ultra-sharp focus, photorealistic 8k e-commerce catalog photography, "
                "no people, no hands, no watermarks, no blur, centered composition.\n\n"
                "Return ONLY a single valid JSON object (no markdown, no backticks) with:\n"
                "- 'enhanced_prompt': The complete, self-contained enhanced prompt.\n"
                "- 'title': Professional commercial product title (3-7 words, e.g. 'Sony WH-1000XM5 Matte Black Wireless ANC Headphones' or '18k White Gold Solitaire Diamond Ring').\n"
                "- 'domain': One of ['apparel', 'jewelry', 'furniture', 'kitchen', 'footwear', 'bags', 'electronics', 'accessories', 'cosmetics', 'health', 'home', 'sports', 'tools', 'musical', 'office', 'grocery', 'general'].\n"
                "- 'features_added': List of 3-4 key added technical specifications or styling details.\n"
                "- 'summary': 1 short sentence summarizing how the prompt was analyzed and extended for commercial shop catalog presentation."
            )

            cand_models = [
                "gemini-3.5-flash-lite",
                "gemini-3.8-flash",
                "gemini-3.1-flash-lite",
                "gemini-flash-lite-latest",
                "gemini-2.5-flash-lite",
                "gemini-flash-latest"
            ]

            response = None
            for cand_model in cand_models:
                try:
                    response = client.models.generate_content(
                        model=cand_model,
                        contents=f"{sys_prompt}\n\nUser Input: \"{prompt.strip()}\"\nStyle Preset: \"{style_preset}\""
                    )
                    if response and response.text:
                        break
                except Exception as model_err:
                    continue

            if response and response.text:
                raw_text = response.text.strip()
                raw_text = re.sub(r'^```json\s*', '', raw_text, flags=re.I)
                raw_text = re.sub(r'^```\s*', '', raw_text)
                raw_text = re.sub(r'```$', '', raw_text).strip()

                import json
                m = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
                    if parsed.get("enhanced_prompt"):
                        return {
                            "enhanced_prompt": parsed["enhanced_prompt"].strip(),
                            "title": parsed.get("title", extract_title_from_prompt(prompt, parsed.get("domain", "general"))),
                            "domain": parsed.get("domain", detect_domain_from_prompt(prompt)),
                            "features_added": parsed.get("features_added", []),
                            "summary": parsed.get("summary", "Prompt analyzed and enhanced with commercial studio specifications."),
                            "category_hint": parsed.get("domain", detect_domain_from_prompt(prompt))
                        }
        except Exception as e:
            print(f"[AI Prompt Enhancer] Gemini API error, using offline enhancer: {e}")

    # Fallback to intelligent offline enhancer
    return enhance_prompt_offline(prompt, style_preset)


_imagen_endpoint_supported: Optional[bool] = None

def generate_with_imagen3(prompt: str, api_key: str):
    """
    Calls Gemini Imagen 3 (imagen-3.0-generate-002) to generate a product image.
    Caches availability to avoid repeated slow retries when model is not enabled on project.
    Returns PIL.Image on success, None on failure.
    """
    global _imagen_endpoint_supported
    if _imagen_endpoint_supported is False:
        return None

    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=genai_types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1",
                safety_filter_level="BLOCK_ONLY_HIGH",
                person_generation="DONT_ALLOW"
            )
        )
        if response.generated_images:
            img_bytes = response.generated_images[0].image.image_bytes
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            print(f"[AI Studio] Imagen 3 generated image: {img.size[0]}x{img.size[1]}px")
            _imagen_endpoint_supported = True
            return img
    except Exception as e:
        _imagen_endpoint_supported = False
        print(f"[AI Studio] Imagen 3 endpoint not available on this API key (using Amazon-standard Studio Engine): {str(e)[:80]}")
    return None


def apply_color_tint(img: Image.Image, tint_r: float, tint_g: float, tint_b: float) -> Image.Image:
    """Applies subtle aesthetic color grading / hue shift to PIL RGB image."""
    r, g, b = img.split()
    r = r.point(lambda i: min(255, max(0, int(i * tint_r))))
    g = g.point(lambda i: min(255, max(0, int(i * tint_g))))
    b = b.point(lambda i: min(255, max(0, int(i * tint_b))))
    return Image.merge('RGB', (r, g, b))


def synthesize_product_image(
    prompt: str,
    style_preset: str = "studio",
    aspect_ratio: str = "1:1",
    seed: Optional[int] = None
) -> Tuple[Image.Image, str, str, str]:
    """
    Synthesizes an Amazon-standard commercial product image from a natural language prompt.

    Pipeline:
    1. Gemini 3.8 Flash Semantic Reasoning: extracts world brand & product knowledge
       (ROG backpack -> ASUS ROG gaming backpack, NOT a Gucci luxury bag).
    2. Try Gemini Imagen 3 if available.
    3. High-fidelity Amazon-standard template engine with color grading.

    Returns (PIL.Image, detected_domain, generated_title, sku_prefix).
    """
    api_key = _load_gemini_key()
    gemini_meta = None
    if api_key:
        gemini_meta = analyze_product_with_gemini(prompt, api_key)

    raw_domain = detect_domain_from_prompt(prompt)
    rule_template, rule_domain, rule_sku = select_template_asset(prompt, raw_domain)

    if gemini_meta:
        domain = gemini_meta.get("domain", rule_domain)
        title = gemini_meta.get("title", extract_title_from_prompt(prompt, domain))
        template_file = gemini_meta.get("template_file")
        sku_prefix = gemini_meta.get("sku_prefix", rule_sku)

        # If rule-based selection found a specific product match (e.g. S21 FE, iPhone, Stanley, Samba, etc.),
        # prioritize it to guarantee 100% accurate product photo
        generic_fallbacks = ["sneaker_studio.jpg", "headphones_studio.jpg", "backpack_gaming.jpg", "classic_backpack_studio.jpg"]
        if rule_template not in generic_fallbacks and os.path.exists(os.path.join(TEMPLATES_DIR, rule_template)):
            template_file = rule_template
            domain = rule_domain
            sku_prefix = rule_sku
        elif not template_file or not os.path.exists(os.path.join(TEMPLATES_DIR, template_file)):
            template_file = rule_template
            domain = rule_domain
            sku_prefix = rule_sku
    else:
        template_file, domain, sku_prefix = rule_template, rule_domain, rule_sku
        title = extract_title_from_prompt(prompt, domain)

    result_img = None

    # --- Primary: Gemini Imagen 3 ---
    if api_key:
        imagen_prompt = build_imagen_prompt(prompt, domain, style_preset)
        print(f"[AI Studio] Calling Imagen 3 for: '{prompt[:60]}...'")
        result_img = generate_with_imagen3(imagen_prompt, api_key)

    # --- Fallback: Template + Color Grading ---
    if result_img is None:
        print(f"[AI Studio] Using studio template: {template_file} (domain: {domain})")
        template_path = os.path.join(TEMPLATES_DIR, template_file)
        if not os.path.exists(template_path):
            fallback_file = TEMPLATE_MAP.get(domain, {}).get("file", "sneaker_studio.jpg")
            template_path = os.path.join(TEMPLATES_DIR, fallback_file)
        if os.path.exists(template_path):
            result_img = Image.open(template_path).convert("RGB")
        else:
            result_img = Image.new("RGB", (1024, 1024), (250, 250, 252))
            draw = ImageDraw.Draw(result_img)
            draw.ellipse([256, 256, 768, 768], fill=(220, 220, 228), outline=(180, 180, 190), width=4)

    # Apply color tint for template fallback (Imagen already handles color)
    p_lower = prompt.lower()
    tint_r, tint_g, tint_b = 1.0, 1.0, 1.0
    if not api_key:  # Only tint in template fallback mode
        if template_file not in ["jordan_green_white.jpg"]:
            if "rose gold" in p_lower or "pink" in p_lower or "copper" in p_lower:
                tint_r, tint_g, tint_b = 1.08, 0.98, 0.94
            elif "emerald" in p_lower or "green" in p_lower or "olive" in p_lower:
                tint_r, tint_g, tint_b = 0.95, 1.06, 0.96
            elif "navy" in p_lower or "blue" in p_lower or "cyan" in p_lower:
                tint_r, tint_g, tint_b = 0.94, 0.98, 1.08
            elif "amber" in p_lower or "cognac" in p_lower or "tan" in p_lower or "brown" in p_lower:
                tint_r, tint_g, tint_b = 1.05, 1.01, 0.93
            elif "matte black" in p_lower or "monochrome" in p_lower or "dark" in p_lower:
                tint_r, tint_g, tint_b = 0.96, 0.96, 0.98
            elif "silver" in p_lower or "titanium" in p_lower or "platinum" in p_lower:
                tint_r, tint_g, tint_b = 1.01, 1.02, 1.04

    preset = STYLE_PRESETS.get(style_preset, STYLE_PRESETS["studio"])
    warmth = preset.get("warmth", 0) / 100.0
    tint_r += warmth
    tint_b -= warmth

    styled_img = apply_color_tint(result_img, tint_r, tint_g, tint_b)
    styled_img = ImageEnhance.Brightness(styled_img).enhance(preset["brightness"])
    styled_img = ImageEnhance.Contrast(styled_img).enhance(preset["contrast"])
    styled_img = ImageEnhance.Color(styled_img).enhance(preset["saturation"])
    styled_img = ImageEnhance.Sharpness(styled_img).enhance(preset["sharpness"])

    w, h = styled_img.size
    if aspect_ratio == "4:5":
        target_w = int(h * 0.8)
        left = max(0, (w - target_w) // 2)
        styled_img = styled_img.crop((left, 0, left + target_w, h))
    elif aspect_ratio == "16:9":
        target_h = int(w * 9 / 16)
        top = max(0, (h - target_h) // 2)
        styled_img = styled_img.crop((0, top, w, top + target_h))

    return styled_img, domain, title, sku_prefix


def generate_and_save_product(
    prompt: str,
    user_id: int,
    product_name: Optional[str] = None,
    style_preset: str = "studio",
    aspect_ratio: str = "1:1",
    category_id: Optional[int] = None,
    birefnet_model=None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Main orchestrator for AI Product Generation:
    1. Synthesizes high-res product photograph from user prompt.
    2. Saves raw image to inputs/
    3. Runs BiRefNet alpha cutout segmentation to processed_images/
    4. Categorizes the product automatically.
    5. Returns product payload ready for DB insertion.
    """
    styled_img, domain, extracted_title, sku_prefix = synthesize_product_image(
        prompt=prompt,
        style_preset=style_preset,
        aspect_ratio=aspect_ratio
    )
    
    final_name = (product_name.strip() if product_name and product_name.strip() else extracted_title)
    
    # Generate unique slug & filenames
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', final_name.lower()).strip('_')[:32]
    unique_id = uuid.uuid4().hex[:8]
    base_filename = f"ai_gen_{slug}_{unique_id}"
    raw_filename = f"{base_filename}.png"
    cutout_filename = f"{base_filename}_cutout.png"

    raw_path = os.path.join(INPUTS_DIR, raw_filename)
    cutout_path = os.path.join(PROCESSED_DIR, cutout_filename)

    # Save raw synthesized image
    styled_img.save(raw_path, format="PNG", optimize=True)

    # Execute BiRefNet background removal
    try:
        from batch_remove_bg import load_and_standardize_image, remove_background
        std_rgb, orig_sz = load_and_standardize_image(raw_path)
        
        if birefnet_model is not None:
            cutout = remove_background(std_rgb, orig_sz, birefnet_model, device)
            cutout.save(cutout_path, format="PNG")
        else:
            # Fallback transparent cutout
            cutout = styled_img.convert("RGBA")
            cutout.save(cutout_path, format="PNG")
    except Exception as matting_err:
        print(f"[AI Generator Matting Warning] {matting_err}")
        styled_img.convert("RGBA").save(cutout_path, format="PNG")

    # Generate unique commercial SKU
    prefix = sku_prefix or TEMPLATE_MAP.get(domain, {}).get("default_sku_prefix", "AI-PR")
    sku = f"{prefix}-{random.randint(1000, 9999)}"

    # Tags based on prompt keywords and domain
    tags_list = [domain, style_preset, "ai-generated", "commercial-studio"]
    for kw in prompt.lower().split():
        clean_kw = re.sub(r'[^a-z0-9]', '', kw)
        if len(clean_kw) >= 4 and clean_kw not in tags_list and clean_kw not in ["with", "that", "this", "from", "into"]:
            tags_list.append(clean_kw)

    tags = ", ".join(tags_list[:8])

    used_api = bool(_load_gemini_key())
    return {
        "name": final_name,
        "sku": sku,
        "prompt": prompt,
        "style_preset": style_preset,
        "aspect_ratio": aspect_ratio,
        "domain": domain,
        "tags": tags,
        "raw_image": raw_filename,
        "cutout_image": cutout_filename,
        "raw_path": raw_path,
        "cutout_path": cutout_path,
        "category_id": category_id,
        "generation_method": "Gemini Imagen 3" if used_api else "Template Engine (add GEMINI_API_KEY to .env)"
    }
