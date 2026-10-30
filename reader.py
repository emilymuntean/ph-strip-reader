"""
reader.py - core pH reading logic (v4: card-based calibration).

Flow:
  1. Gemini locates strip boxes, reference-card swatch boxes (+ pH labels), and a white patch.
  2. White balance the whole image using the white patch.
  3. Build a per-photo calibration chart by sampling each swatch's Lab color.
  4. For each strip, sample its Lab color and project onto the swatch curve to get pH.
"""
import base64
import io
import json
import os

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- tunables ----
GAMMA = 2.2
WHITE_TARGET = 235.0
PAD_SHRINK = 0.15
DIST_HIGH = float(os.environ.get("DIST_HIGH", 8.0))
DIST_MED = float(os.environ.get("DIST_MED", 15.0))
GLARE_L_MAX = 93.0
GLARE_C_MAX = 10.0

# ---- Gemini setup ----
_genai_ready = False
_vision = None

def _init_genai():
    global _genai_ready, _vision
    if _genai_ready:
        return _vision
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)
    _vision = genai.GenerativeModel(
        os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    )
    _genai_ready = True
    return _vision


LOCATE_PROMPT = """You locate objects in a photo of a foot mat used for diabetic foot ulcer screening.

WHAT IS IN THE PHOTO
- Two foot outlines (left foot, right foot).
- 8 small pH test strips (4 per foot) sitting inside slots. Each strip has a dry bright-yellow tail and a wet/reacted colored pad (green, olive, orange, brown, etc.).
- ONE printed reference color card somewhere in the photo, showing labeled pH color swatches. The pH value of each swatch is printed next to it.
- A plain white area on the mat or paper (the white border of an alignment tile is usually fine).

WHAT TO RETURN (valid JSON only, no markdown):

{
  "strips": [
    {"zone_id": "zone_01", "bbox": [left, top, right, bottom]},
    {"zone_id": "zone_02", "bbox": [...]},
    ... EXACTLY 8 entries ...
  ],
  "swatches": [
    {"ph": 6.0, "bbox": [left, top, right, bottom]},
    {"ph": 6.2, "bbox": [...]},
    ... one entry per visible labeled swatch on the card ...
  ],
  "white_box": [left, top, right, bottom],
  "needs_retake": false,
  "reason": ""
}

RULES
- Coordinates are normalized 0..1, origin top-left, bbox = [left, top, right, bottom].
- For each strip, box ONLY the wet/reacted colored end. Exclude the dry yellow tail and any tape.
- For each swatch, box the colored square. Read the printed pH number next to it and put it in "ph" as a number.
- white_box must be a plain white region with no text, no swatch color, no shadow, no glare.
- If you cannot confidently locate all 8 strips or at least 3 swatches, set needs_retake=true and explain in "reason".
- Do NOT invent pads or swatches. Return only what you can see.
"""


# ------------------------------------------------------------------ color helpers
def rgb01_to_lab(rgb01):
    a = np.asarray(rgb01, np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)


def classify(ph):
    if ph < 6.5:
        return "acidic"
    if ph <= 7.5:
        return "neutral"
    return "alkaline"


# ------------------------------------------------------------------ Gemini locate
def locate_all(img_bgr, timeout_s=30):
    """Call Gemini to locate strips, swatches (with pH labels), and a white box."""
    model = _init_genai()
    
    # --- NEW: Resize the image to save memory and speed up Gemini ---
    h, w = img_bgr.shape[:2]
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # ----------------------------------------------------------------

    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Could not encode image to JPEG")
    pil = Image.open(io.BytesIO(buf.tobytes()))
    resp = model.generate_content([LOCATE_PROMPT, pil])
    txt = (resp.text or "").strip()
    txt = txt.replace("```json", "").replace("```", "").strip()
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Gemini returned non-JSON: {txt[:300]}")
    return json.loads(txt[start:end + 1])


# ------------------------------------------------------------------ white balance
def white_balance_with_box(img_bgr, box_norm):
    """White-balance the whole image in linear light using the given normalized box."""
    h, w = img_bgr.shape[:2]
    l, t, r, b = box_norm
    x1 = max(0, int(l * w)); y1 = max(0, int(t * h))
    x2 = min(w, int(r * w)); y2 = min(h, int(b * h))
    if x2 - x1 < 5 or y2 - y1 < 5:
        raise ValueError("white_box too small or out of bounds")

    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    lin = rgb ** GAMMA
    patch = lin[y1:y2, x1:x2].reshape(-1, 3)
    ref = np.median(patch, axis=0)
    target = (WHITE_TARGET / 255.0) ** GAMMA
    gains = target / np.maximum(ref, 1e-4)
    out = np.clip(lin * gains, 0, 1) ** (1.0 / GAMMA)

    # stats for warnings
    white_srgb = (ref ** (1.0 / GAMMA)) * 255.0
    return out, {
        "white_srgb": [round(float(v), 1) for v in white_srgb],
        "gains": [round(float(v), 3) for v in gains],
    }


# ------------------------------------------------------------------ sampling
def sample_box_lab(bal01, box_norm):
    """Sample median Lab color from a normalized box in a white-balanced image."""
    h, w = bal01.shape[:2]
    l, t, r, b = box_norm
    x1 = max(0, int(l * w)); y1 = max(0, int(t * h))
    x2 = min(w, int(r * w)); y2 = min(h, int(b * h))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None, None

    dx = int((x2 - x1) * PAD_SHRINK)
    dy = int((y2 - y1) * PAD_SHRINK)
    roi = bal01[y1 + dy:y2 - dy, x1 + dx:x2 - dx].reshape(-1, 3)
    if roi.shape[0] < 20:
        return None, None

    lab = rgb01_to_lab(roi)
    C = np.hypot(lab[:, 1], lab[:, 2])
    glare = (lab[:, 0] > GLARE_L_MAX) & (C < GLARE_C_MAX)
    glare_frac = float(glare.mean())
    if (~glare).sum() >= 20:
        lab = lab[~glare]
        roi = roi[~glare]
    med = np.median(lab, axis=0)
    spread = float(np.mean(np.linalg.norm(lab - med, axis=1)))
    return med, {"spread": spread, "glare_frac": glare_frac,
                 "rgb": (np.median(roi, axis=0) * 255).round().astype(int).tolist()}


# ------------------------------------------------------------------ calibration from card
def build_chart_from_card(bal01, swatches):
    """Build {ph: Lab} by sampling each swatch on the reference card."""
    chart = {}
    for sw in swatches:
        try:
            ph = float(sw["ph"])
            bbox = sw["bbox"]
        except (KeyError, TypeError, ValueError):
            continue
        lab, _ = sample_box_lab(bal01, bbox)
        if lab is not None:
            chart[ph] = lab
    return dict(sorted(chart.items()))


# ------------------------------------------------------------------ Lab -> pH
def lab_to_ph(lab, chart):
    """Project onto the piecewise-linear swatch curve in Lab space."""
    keys = list(chart)
    best = (1e9, keys[0], 0, 0.0)
    for i in range(len(keys) - 1):
        p0, p1 = chart[keys[i]], chart[keys[i + 1]]
        seg = p1 - p0
        denom = max(float(np.dot(seg, seg)), 1e-9)
        t = float(np.clip(np.dot(lab - p0, seg) / denom, 0, 1))
        d = float(np.linalg.norm(lab - (p0 + t * seg)))
        if d < best[0]:
            best = (d, keys[i] + t * (keys[i + 1] - keys[i]), i, t)
    d, ph, i, t = best
    flag = None
    if i == 0 and t <= 0.02:
        flag = "below"
    elif i == len(keys) - 2 and t >= 0.98:
        flag = "above"
    return round(float(ph), 2), round(float(d), 1), flag


# ------------------------------------------------------------------ full analyze
def analyze(img_bgr, rec_id=None):
    loc = locate_all(img_bgr)
    if loc.get("needs_retake"):
        return {
            "id": rec_id,
            "error": "Retake photo: " + (loc.get("reason") or "unreadable"),
            "needs_retake": True,
            "strips": [],
            "swatches_found": 0,
        }

    strips_in = loc.get("strips") or []
    swatches_in = loc.get("swatches") or []
    white_box = loc.get("white_box")

    if not white_box:
        return {"id": rec_id, "error": "No white reference box found",
                "needs_retake": True, "strips": [], "swatches_found": 0}

    try:
        wb, wb_info = white_balance_with_box(img_bgr, white_box)
    except Exception as e:
        return {"id": rec_id, "error": f"White balance failed: {e}",
                "needs_retake": True, "strips": [], "swatches_found": 0}

    chart = build_chart_from_card(wb, swatches_in)
    if len(chart) < 3:
        return {
            "id": rec_id,
            "error": f"Only {len(chart)} swatches readable from reference card (need >=3)",
            "needs_retake": True,
            "strips": [],
            "swatches_found": len(chart),
            "white_srgb": wb_info["white_srgb"],
        }

    chart_range = (min(chart), max(chart))
    strips_out = []
    for i, s in enumerate(strips_in, 1):
        rec = {
            "strip_id": s.get("zone_id") or f"strip_{i}",
            "ph": None,
            "classification": None,
            "confidence": "none",
            "color_distance": None,
            "range_flag": None,
            "status": "needs_review",
            "note": "",
        }
        lab, meta = sample_box_lab(wb, s.get("bbox") or [0, 0, 0, 0])
        if lab is None:
            rec["note"] = "strip box unreadable"
            strips_out.append(rec)
            continue

        ph, dist, flag = lab_to_ph(lab, chart)
        rec["ph"] = ph
        rec["classification"] = classify(ph)
        rec["color_distance"] = dist
        rec["range_flag"] = flag

        if dist <= DIST_HIGH:
            rec["confidence"] = "high"
        elif dist <= DIST_MED:
            rec["confidence"] = "medium"
        else:
            rec["confidence"] = "low"

        notes = []
        if meta and meta.get("spread", 0) > 12:
            notes.append("uneven color across pad")
        if meta and meta.get("glare_frac", 0) > 0.30:
            notes.append("glare on pad")
        if flag == "below":
            notes.append(f"below card range (<= {chart_range[0]:g})")
        elif flag == "above":
            notes.append(f"above card range (>= {chart_range[1]:g})")
        if rec["confidence"] == "low":
            notes.append("color does not match any swatch well")

        if rec["confidence"] in ("high", "medium") and not notes:
            rec["status"] = "ok"
        rec["note"] = "; ".join(notes)
        strips_out.append(rec)

    return {
        "id": rec_id,
        "num_strips": len(strips_out),
        "swatches_found": len(chart),
        "swatch_phs": sorted(chart.keys()),
        "white_srgb": wb_info["white_srgb"],
        "wb_gains": wb_info["gains"],
        "strips": strips_out,
        "_wb_image": wb,
        "_locate": loc,
    }


# ------------------------------------------------------------------ drawing
def annotate(img_bgr, loc):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    th = max(2, w // 600)
    fs = max(0.5, w / 1600)

    def draw(box, color, label):
        l, t, r, b = box
        x1, y1 = int(l * w), int(t * h)
        x2, y2 = int(r * w), int(b * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, th)
        if label:
            cv2.putText(out, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, th)

    for s in loc.get("strips") or []:
        draw(s.get("bbox"), (0, 200, 0), s.get("zone_id", ""))
    for sw in loc.get("swatches") or []:
        draw(sw.get("bbox"), (255, 0, 0), f"pH {sw.get('ph','?')}")
    if loc.get("white_box"):
        draw(loc["white_box"], (255, 0, 255), "WB")
    return out


def encode_b64(img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()
