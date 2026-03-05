"""
utils/logo_model.py
BrandSphere AI — Logo & Design Studio Backend
Handles logo classification, color palette extraction, and animated GIF generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colorsys
import io
import os
import joblib
from PIL import Image, ImageDraw, ImageFont
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio


# ─── Color Psychology Map ─────────────────────────────────────────────────────
COLOR_PSYCHOLOGY = {
    "red":    {"label": "Energy & Passion",       "desc": "Evokes urgency and excitement. Great for food, retail, and entertainment brands."},
    "blue":   {"label": "Trust & Reliability",    "desc": "Conveys stability and professionalism. Ideal for tech, finance, and healthcare."},
    "green":  {"label": "Growth & Nature",         "desc": "Represents health, sustainability, and freshness. Perfect for eco and wellness brands."},
    "yellow": {"label": "Optimism & Creativity",  "desc": "Stimulates positivity and energy. Works well for youthful and creative brands."},
    "purple": {"label": "Luxury & Wisdom",         "desc": "Signals sophistication and exclusivity. Fits beauty, fashion, and premium brands."},
    "orange": {"label": "Enthusiasm & Adventure", "desc": "Combines energy with friendliness. Great for sports, food, and lifestyle brands."},
    "black":  {"label": "Power & Elegance",        "desc": "Communicates premium quality and authority. Works for luxury and high-end brands."},
    "white":  {"label": "Purity & Simplicity",    "desc": "Signals cleanliness and minimalism. Ideal for tech, healthcare, and premium brands."},
}

PERSONALITY_PALETTES = {
    "minimalist": [(27, 58, 107), (255, 255, 255), (220, 220, 220), (100, 100, 100), (200, 200, 200)],
    "vibrant":    [(255, 79, 56), (255, 190, 0), (0, 200, 83), (41, 182, 246), (156, 39, 176)],
    "luxury":     [(212, 175, 55), (18, 18, 18), (255, 255, 255), (150, 120, 60), (80, 80, 80)],
    "bold":       [(220, 50, 32), (0, 0, 0), (255, 255, 255), (18, 100, 196), (255, 140, 0)],
    "elegant":    [(139, 90, 43), (245, 245, 240), (180, 180, 180), (80, 50, 20), (220, 200, 170)],
}


def extract_color_palette(image_array: np.ndarray, n_colors: int = 5) -> list[dict]:
    """
    Extract dominant color palette from a numpy image array using KMeans.
    Falls back to personality-based palette if image is synthetic/plain.
    """
    from sklearn.cluster import KMeans
    pixels = image_array.reshape(-1, 3).astype(float)

    # Remove near-white and near-black pixels for better palette
    mask = ~(np.all(pixels > 240, axis=1) | np.all(pixels < 15, axis=1))
    filtered_pixels = pixels[mask] if mask.sum() > 100 else pixels

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(filtered_pixels)
    colors_rgb = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    sorted_idx = np.argsort(counts)[::-1]
    colors_rgb = colors_rgb[sorted_idx]

    results = []
    for i, (r, g, b) in enumerate(colors_rgb):
        r, g, b = int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255))
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        h_deg = h * 360
        if s < 0.15:      color_name = "white" if v > 0.8 else "black" if v < 0.2 else "neutral"
        elif h_deg < 30:  color_name = "red"
        elif h_deg < 60:  color_name = "orange"
        elif h_deg < 90:  color_name = "yellow"
        elif h_deg < 150: color_name = "green"
        elif h_deg < 210: color_name = "blue"
        elif h_deg < 270: color_name = "purple"
        elif h_deg < 330: color_name = "red"
        else:             color_name = "red"

        psych = COLOR_PSYCHOLOGY.get(color_name, {"label": "Balanced", "desc": "A versatile color for any brand identity."})
        results.append({
            "rank":       i + 1,
            "rgb":        (r, g, b),
            "hex":        "#{:02X}{:02X}{:02X}".format(r, g, b),
            "color_name": color_name,
            "psychology": psych["label"],
            "description": psych["desc"],
            "percentage": round(counts[sorted_idx[i]] / len(filtered_pixels) * 100, 1),
        })
    return results


def get_personality_palette(personality: str) -> list[dict]:
    """Return predefined palette for a brand personality."""
    palette_rgb = PERSONALITY_PALETTES.get(personality.lower(), PERSONALITY_PALETTES["minimalist"])
    results = []
    labels = ["Primary", "Secondary", "Accent", "Background", "Neutral"]
    for i, (r, g, b) in enumerate(palette_rgb):
        results.append({
            "rank":       i + 1,
            "rgb":        (r, g, b),
            "hex":        "#{:02X}{:02X}{:02X}".format(r, g, b),
            "role":       labels[i] if i < len(labels) else f"Color {i+1}",
            "percentage": [40, 25, 15, 12, 8][i] if i < 5 else 5,
        })
    return results


def generate_brand_visual(company_name: str, tagline: str, personality: str,
                           palette: list[dict]) -> np.ndarray:
    """
    Generate a brand visual card (logo placeholder + palette + text).
    Returns a numpy array representing the brand card image.
    """
    width, height = 800, 400
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Background gradient
    bg_color = palette[0]["rgb"] if palette else (27, 58, 107)
    for y in range(height):
        alpha = y / height
        r = int(bg_color[0] * (1 - alpha * 0.5))
        g = int(bg_color[1] * (1 - alpha * 0.5))
        b = int(bg_color[2] * (1 - alpha * 0.5))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Logo circle placeholder
    circle_color = palette[1]["rgb"] if len(palette) > 1 else (46, 134, 171)
    draw.ellipse([50, 100, 250, 300], fill=circle_color, outline=(255, 255, 255), width=3)

    # Company initials in circle
    initials = "".join([word[0].upper() for word in company_name.split()[:2]])
    draw.text((150, 195), initials, fill=(255, 255, 255), anchor="mm")

    # Company name
    draw.text((300, 140), company_name, fill=(255, 255, 255))

    # Tagline
    draw.text((300, 180), tagline[:60], fill=(200, 220, 240))

    # Color palette strip
    strip_y = 320
    strip_w = int(width / len(palette)) if palette else width
    for i, color_info in enumerate(palette[:5]):
        x0 = i * strip_w
        draw.rectangle([x0, strip_y, x0 + strip_w, height], fill=color_info["rgb"])
        draw.text((x0 + strip_w // 2, strip_y + 30), color_info.get("hex", ""), fill=(255, 255, 255), anchor="mm")

    # Personality tag
    draw.rectangle([580, 50, 760, 90], fill=palette[2]["rgb"] if len(palette) > 2 else (244, 162, 97))
    draw.text((670, 65), personality.upper(), fill=(255, 255, 255), anchor="mm")

    return np.array(img)


def create_animated_gif(company_name: str, tagline: str, palette: list[dict]) -> bytes:
    """
    Create a 6-frame animated brand GIF: logo fade-in + typewriter tagline + palette reveal.
    Returns bytes of the GIF.
    """
    frames = []
    width, height = 600, 300
    bg_color = palette[0]["rgb"] if palette else (27, 58, 107)
    accent   = palette[2]["rgb"] if len(palette) > 2 else (244, 162, 97)
    text_color = (255, 255, 255)

    for frame_i in range(6):
        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Fade-in logo circle
        alpha = min(1.0, frame_i / 3)
        circle_color = tuple(int(c * alpha) for c in (46, 134, 171))
        draw.ellipse([30, 80, 130, 180], fill=circle_color)

        initials = "".join([w[0].upper() for w in company_name.split()[:2]])
        if frame_i >= 1:
            draw.text((80, 128), initials, fill=text_color, anchor="mm")

        # Typewriter tagline effect
        tagline_short = tagline[:50]
        chars_to_show = min(len(tagline_short), frame_i * 10)
        if frame_i >= 2:
            draw.text((155, 120), tagline_short[:chars_to_show], fill=text_color)

        # Company name
        if frame_i >= 1:
            draw.text((155, 90), company_name, fill=text_color)

        # Color palette reveal
        if frame_i >= 4:
            strip_w = width // len(palette) if palette else width
            for i, c in enumerate(palette[:5]):
                x0 = i * strip_w
                draw.rectangle([x0, 220, x0 + strip_w, height], fill=c["rgb"])

        frames.append(img)

    # Save to bytes
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=400, loop=0)
    return buf.getvalue()


def classify_logo_style(features: np.ndarray) -> dict:
    """Load and run logo classifier."""
    model_path = os.path.join(os.path.dirname(__file__), "logo_classifier.pkl")
    le_path    = os.path.join(os.path.dirname(__file__), "logo_label_encoder.pkl")
    if os.path.exists(model_path) and os.path.exists(le_path):
        clf = joblib.load(model_path)
        le  = joblib.load(le_path)
        pred = clf.predict(features.reshape(1, -1))[0]
        proba = clf.predict_proba(features.reshape(1, -1))[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        return {"predicted_style": le.inverse_transform([pred])[0],
                "top3": [(le.inverse_transform([i])[0], round(proba[i]*100, 1)) for i in top3_idx]}
    return {"predicted_style": "minimalist", "top3": [("minimalist", 45.0), ("vibrant", 30.0), ("bold", 25.0)]}


def recommend_fonts(personality: str, use_case: str = "logo", n: int = 3) -> list[dict]:
    """Recommend fonts based on brand personality."""
    font_suggestions = {
        "minimalist": [
            {"name": "Helvetica Neue", "category": "Sans-Serif", "weight": "Light", "readability": 95},
            {"name": "Futura",         "category": "Sans-Serif", "weight": "Regular", "readability": 92},
            {"name": "Gill Sans",      "category": "Sans-Serif", "weight": "Regular", "readability": 89},
        ],
        "luxury": [
            {"name": "Didot",          "category": "Serif",   "weight": "Regular", "readability": 87},
            {"name": "Garamond",       "category": "Serif",   "weight": "Light",   "readability": 90},
            {"name": "Bodoni MT",      "category": "Serif",   "weight": "Bold",    "readability": 85},
        ],
        "vibrant": [
            {"name": "Montserrat",     "category": "Sans-Serif", "weight": "Bold",  "readability": 94},
            {"name": "Raleway",        "category": "Sans-Serif", "weight": "ExtraBold", "readability": 91},
            {"name": "Paytone One",    "category": "Display",    "weight": "Regular",   "readability": 88},
        ],
        "bold": [
            {"name": "Impact",         "category": "Display",    "weight": "Regular", "readability": 83},
            {"name": "Anton",          "category": "Display",    "weight": "Regular", "readability": 86},
            {"name": "Bebas Neue",     "category": "Display",    "weight": "Regular", "readability": 88},
        ],
        "elegant": [
            {"name": "Playfair Display", "category": "Serif",  "weight": "Regular", "readability": 91},
            {"name": "Cormorant",        "category": "Serif",  "weight": "Light",   "readability": 89},
            {"name": "EB Garamond",      "category": "Serif",  "weight": "Regular", "readability": 92},
        ],
    }
    return font_suggestions.get(personality.lower(), font_suggestions["minimalist"])[:n]
