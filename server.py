"""
pH Strip Reader API  (v2 - for the DFU pH mat workflow)
-------------------------------------------------------
POST /analyze  (multipart form)
  file       : the mat photo (required)
  boxes      : optional JSON. Pad locations, normalized 0-1, [left, top, right, bottom].
               Accepts either [[l,t,r,b], ...] or {"zones":[{"zone_id":..,"bbox":[..]}], "white_box":[..]}
               (this is exactly what the n8n "locate pH strips" node returns)
  white_box  : optional JSON [l,t,r,b] of a plain white area (mat background / white card)
  expected   : optional int, number of pads expected (e.g. 4)
  id         : optional, passed through to the output

The pH number is ALWAYS computed here with color math (CIELAB + calibration curve).
No AI model ever decides the pH value.
"""
import base64, json, os
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "best.pt")
CHART_PATH = os.path.join(HERE, "chart.json")

# Placeholder chart (pH 3-10). NOT accurate for your strips -> run calibrate.py and upload chart.json.
GENERIC_CHART_RGB = {
    3.0: (245, 120, 40), 4.0: (245, 170, 40), 5.0: (240, 215, 50), 6.0: (190, 210, 60),
    7.0: (90, 170, 70), 8.0: (40, 150, 120), 9.0: (40, 110, 170), 10.0: (60, 70, 160),
}


def rgb_to_lab(rgb):
    px = np.uint8([[np.clip(np.round(rgb), 0, 255)]])
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    return np.array([lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128])


def load_chart():
    if os.path.exists(CHART_PATH):
        with open(CHART_PATH) as f:
            data = json.load(f)
        chart = {float(k): np.array(v, float) for k, v in data.items()}
        return dict(sorted(chart.items())), "calibrated"
    return {k: rgb_to_lab(v) for k, v in sorted(GENERIC_CHART_RGB.items())}, "generic (NOT calibrated)"


CHART, CHART_SOURCE = load_chart()

YOLO_MODEL = None
if os.path.exists(MODEL_PATH):
    from ultralytics import YOLO
    YOLO_MODEL = YOLO(MODEL_PATH)


# ---------------- helpers ----------------
def to_px(box, w, h):
    """normalized [l,t,r,b] -> pixel ints; passes pixel boxes through."""
    l, t, r, b = [float(v) for v in box]
    if max(l, t, r, b) <= 1.5:
        l, r, t, b = l * w, r * w, t * h, b * h
    l, r = sorted((int(round(l)), int(round(r))))
    t, b = sorted((int(round(t)), int(round(b))))
    return [max(0, l), max(0, t), min(w, r), min(h, b)]


def parse_boxes(raw):
    """Returns (list_of_boxes, white_box_or_None)."""
    if not raw:
        return None, None
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, dict):
        zones = data.get("zones") or []
        boxes = [z["bbox"] if isinstance(z, dict) else z for z in zones]
        return boxes, data.get("white_box")
    return data, None


def sort_reading_order(boxes):
    """Top-to-bottom, then left-to-right (rows grouped by vertical overlap)."""
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    rows, cur = [], []
    for b in boxes:
        if cur and (b[1] + b[3]) / 2 > np.mean([(c[1] + c[3]) / 2 for c in cur]) + \
                0.5 * np.mean([c[3] - c[1] for c in cur]):
            rows.append(cur); cur = []
        cur.append(b)
    if cur:
        rows.append(cur)
    return [b for row in rows for b in sorted(row, key=lambda b: b[0])]


def detect_opencv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 70, 50), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = img.shape[0] * img.shape[1] * 0.002
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            out.append([x, y, x + w, y + h])
    return out


def white_balance(img, white_px_box=None):
    """Scale channels so a known-white area becomes neutral. Falls back to gray-world."""
    f = img.astype(np.float32)
    if white_px_box:
        x1, y1, x2, y2 = white_px_box
        ref = np.median(f[y1:y2, x1:x2].reshape(-1, 3), axis=0)
        target = 240.0
    else:
        ref = f.reshape(-1, 3).mean(0)
        target = ref.mean()
    f *= target / np.maximum(ref, 1)
    return np.clip(f, 0, 255).astype(np.uint8)


def pad_color(img, box):
    """Median RGB of the center 50% of the pad (avoids edges/glare)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad = img[y1 + h // 4: y2 - h // 4, x1 + w // 4: x2 - w // 4]
    if pad.size == 0:
        pad = img[y1:y2, x1:x2]
    px = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(float)
    # drop glare (very bright, low-saturation pixels)
    keep = ~((px.min(1) > 235))
    if keep.sum() > 20:
        px = px[keep]
    rgb = np.median(px, axis=0)
    spread = float(np.std(px, axis=0).mean())  # high = uneven / bleeding pad
    return rgb, spread


def lab_to_ph(lab):
    """Project the color onto the piecewise-linear calibration curve in Lab space.
    Gives a continuous pH between chart points (needed for +/-0.2 precision)."""
    keys = list(CHART)
    best = (1e9, keys[0])
    for k0, k1 in zip(keys[:-1], keys[1:]):
        p0, p1 = CHART[k0], CHART[k1]
        seg = p1 - p0
        t = float(np.clip(np.dot(lab - p0, seg) / max(np.dot(seg, seg), 1e-9), 0, 1))
        d = float(np.linalg.norm(lab - (p0 + t * seg)))
        if d < best[0]:
            best = (d, k0 + t * (k1 - k0))
    return round(best[1], 2), round(best[0], 1)


def classify(ph):
    if ph < 6.5:
        return "acidic"
    if ph <= 7.5:
        return "neutral"
    return "alkaline"


def analyze_image(img, boxes=None, white_box=None, expected=None, rec_id=None):
    h, w = img.shape[:2]
    if boxes:
        px_boxes, detector = [to_px(b, w, h) for b in boxes], "provided-boxes"
    elif YOLO_MODEL:
        res = YOLO_MODEL(img, verbose=False)[0]
        px_boxes, detector = res.boxes.xyxy.int().tolist(), "yolo"
    else:
        px_boxes, detector = detect_opencv(img), "opencv-fallback"
    px_boxes = [b for b in px_boxes if b[2] - b[0] > 3 and b[3] - b[1] > 3]
    px_boxes = sort_reading_order(px_boxes)

    wb = to_px(white_box, w, h) if white_box else None
    balanced = white_balance(img, wb)

    count_ok = expected is None or len(px_boxes) == expected
    strips, annotated = [], img.copy()
    for n, box in enumerate(px_boxes, 1):
        rgb, spread = pad_color(balanced, box)
        ph, dist = lab_to_ph(rgb_to_lab(rgb))
        if dist <= 8 and spread <= 20:
            conf = "high"
        elif dist <= 15:
            conf = "medium"
        else:
            conf = "low"
        status = "ok" if (conf != "low" and count_ok) else "needs_review"
        strips.append({
            "id": rec_id, "strip_id": f"strip_{n}", "ph": ph, "classification": classify(ph),
            "confidence": conf, "color_distance": dist, "status": status,
            "box": [int(v) for v in box], "color_rgb": [int(v) for v in rgb],
        })
        x1, y1, x2, y2 = box
        color = (0, 200, 0) if status == "ok" else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"{n}: pH {ph:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 12)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    if wb:
        cv2.rectangle(annotated, (wb[0], wb[1]), (wb[2], wb[3]), (255, 255, 255), 2)
    return {
        "id": rec_id, "num_strips": len(strips), "expected": expected, "count_ok": count_ok,
        "detector": detector, "chart": CHART_SOURCE,
        "white_balance": "white_box" if wb else "gray-world",
        "strips": strips, "_annotated": annotated,
    }


app = FastAPI(title="pH Strip Reader")


@app.get("/health")
def health():
    return {"ok": True, "detector": "yolo" if YOLO_MODEL else "opencv-fallback (or provided boxes)",
            "chart": CHART_SOURCE, "chart_points": list(CHART)}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), boxes: str = Form(None), white_box: str = Form(None),
                  expected: int = Form(None), id: str = Form(None)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not read image"}
    try:
        bx, wb_from_boxes = parse_boxes(boxes)
        wb = json.loads(white_box) if white_box else wb_from_boxes
    except Exception as e:
        return {"error": f"Could not parse boxes/white_box JSON: {e}"}
    result = analyze_image(img, bx, wb, expected, id)
    ok, buf = cv2.imencode(".jpg", result.pop("_annotated"))
    result["annotated_image_base64"] = base64.b64encode(buf).decode()
    return result
