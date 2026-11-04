import base64
import json
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

HERE = os.path.dirname(os.path.abspath(__file__))
CHART_PATH = os.path.join(HERE, "chart.json")

def load_chart():
    if os.path.exists(CHART_PATH):
        with open(CHART_PATH) as f:
            data = json.load(f)
        chart = {float(k): np.array(v, float) for k, v in data.items()}
        return dict(sorted(chart.items())), "calibrated"
    return None, "none"

CHART, CHART_SOURCE = load_chart()

def rgb_to_lab(rgb):
    px = np.uint8([[np.clip(np.round(rgb), 0, 255)]])
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    return np.array([lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128])

def to_px(box, w, h):
    l, t, r, b = [float(v) for v in box]
    if max(l, t, r, b) <= 1.5:
        l, r, t, b = l * w, r * w, t * h, b * h
    l, r = sorted((int(round(l)), int(round(r))))
    t, b = sorted((int(round(t)), int(round(b))))
    return [max(0, l), max(0, t), min(w, r), min(h, b)]

def parse_boxes(raw):
    if not raw: return None, None
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, dict):
        zones = data.get("zones") or data.get("strips") or []
        boxes = [z["bbox"] if isinstance(z, dict) else z for z in zones]
        return boxes, data.get("white_box")
    return data, None

def white_balance(img, white_px_box=None):
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
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad = img[y1 + h // 4: y2 - h // 4, x1 + w // 4: x2 - w // 4]
    if pad.size == 0:
        pad = img[y1:y2, x1:x2]
    px = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(float)
    keep = ~((px.min(1) > 235)) # drop glare
    if keep.sum() > 20:
        px = px[keep]
    rgb = np.median(px, axis=0)
    return rgb

def lab_to_ph(lab, chart=None):
    chart = chart or CHART
    if not chart: return None, None
    keys = list(chart)
    best = (1e9, keys[0])
    for k0, k1 in zip(keys[:-1], keys[1:]):
        p0, p1 = chart[k0], chart[k1]
        seg = p1 - p0
        t = float(np.clip(np.dot(lab - p0, seg) / max(np.dot(seg, seg), 1e-9), 0, 1))
        d = float(np.linalg.norm(lab - (p0 + t * seg)))
        if d < best[0]:
            best = (d, k0 + t * (k1 - k0))
    return round(best[1], 2), round(best[0], 1)

def classify(ph):
    if ph < 6.5: return "acidic"
    if ph <= 7.5: return "neutral"
    return "alkaline"

def analyze_image(img, boxes=None, white_box=None, rec_id=None):
    h, w = img.shape[:2]
    if boxes:
        px_boxes = [to_px(b, w, h) for b in boxes]
        detector = "provided-boxes"
    else:
        # Fallback to Emily's OpenCV detector if no boxes provided
        from server import detect_mat # type: ignore
        px_boxes, _, _, _, _, _ = detect_mat(img)
        detector = "opencv-detector"
        
    wb = to_px(white_box, w, h) if white_box else None
    balanced = white_balance(img, wb)
    
    strips = []
    for n, box in enumerate(px_boxes, 1):
        rgb = pad_color(balanced, box)
        ph, dist = lab_to_ph(rgb_to_lab(rgb))
        
        conf = "high" if dist is not None and dist <= 8 else ("medium" if dist is not None and dist <= 15 else "low")
        if ph is None: conf = "none"
        
        strips.append({
            "strip_id": f"strip_{n}", "ph": ph,
            "classification": classify(ph) if ph is not None else None,
            "confidence": conf, "color_distance": dist,
            "status": "ok" if conf in ("high", "medium") else "needs_review",
            "box": [int(v) for v in box]
        })
        
    return {
        "id": rec_id, "num_strips": len(strips), "detector": detector,
        "chart": CHART_SOURCE, "strips": strips
    }

app = FastAPI(title="pH Reader Combined")

@app.get("/health")
def health():
    return {"ok": True, "chart_source": CHART_SOURCE, "points": sorted(CHART.keys()) if CHART else []}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), boxes: str = Form(None), id: str = Form(None)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: return {"error": "Could not read image"}
    try:
        bx, wb = parse_boxes(boxes)
    except Exception as e:
        return {"error": f"Could not parse boxes: {e}"}
    return analyze_image(img, bx, wb, id)
