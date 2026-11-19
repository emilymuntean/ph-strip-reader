import base64
import json
import os
import glob
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

HERE = os.path.dirname(os.path.abspath(__file__))
CHART_PATH = os.path.join(HERE, "chart.json")
CALIB_DIR = os.path.join(HERE, "calib")

# ---------------- Calibration Auto-Generation ----------------
def rgb_to_lab(rgb):
    px = np.uint8([[np.clip(np.round(rgb), 0, 255)]])
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    return np.array([lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128])

def auto_generate_chart():
    if not os.path.exists(CALIB_DIR):
        return None
    
    chart = {}
    for ph_dir in sorted(os.listdir(CALIB_DIR)):
        try:
            ph_val = float(ph_dir)
        except ValueError:
            continue
            
        folder_path = os.path.join(CALIB_DIR, ph_dir)
        if not os.path.isdir(folder_path):
            continue
            
        labs = []
        for img_path in glob.glob(os.path.join(folder_path, "*.*")):
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Auto White Balance (brightest 1%)
            f = img.astype(np.float32)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bright_idx = np.argsort(gray.flatten())[-int(gray.size * 0.01):]
            ref = np.median(f.reshape(-1, 3)[bright_idx], axis=0)
            f *= 240.0 / np.maximum(ref, 1)
            balanced = np.clip(f, 0, 255).astype(np.uint8)
            
            # Auto Pad Detection (most saturated blob)
            hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            pad = balanced[y + h//4: y + h - h//4, x + w//4: x + w - w//4]
            if pad.size == 0: pad = balanced[y:y+h, x:x+w]
            
            px = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(float)
            rgb = np.median(px, axis=0)
            labs.append(rgb_to_lab(rgb))
            
        if labs:
            mean_lab = np.mean(labs, axis=0).round(2).tolist()
            chart[str(ph_val)] = mean_lab

    if chart:
        print(f"Generated chart with {len(chart)} points from calib/ folder.")
        with open(CHART_PATH, "w") as f:
            json.dump(chart, f, indent=2)
        return dict(sorted(chart.items()))
    return None

def load_chart():
    if os.path.exists(CHART_PATH):
        with open(CHART_PATH) as f:
            data = json.load(f)
        chart = {float(k): np.array(v, float) for k, v in data.items()}
        return dict(sorted(chart.items())), "chart.json"
    
    chart = auto_generate_chart()
    if chart:
        return chart, "auto-generated from photos"
        
    return None, "none"

CHART, CHART_SOURCE = load_chart()

# ---------------- Image Processing Helpers ----------------
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
    if pad.size == 0: pad = img[y1:y2, x1:x2]
    px = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(float)
    keep = ~((px.min(1) > 235))
    if keep.sum() > 20: px = px[keep]
    return np.median(px, axis=0)

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

# ---------------- Main Analysis ----------------
def analyze_image(img, boxes=None, white_box=None, rec_id=None):
    h, w = img.shape[:2]
    
    if boxes:
        px_boxes = [to_px(b, w, h) for b in boxes]
        detector = "provided-boxes"
    else:
        return {"error": "No boxes provided from n8n"}
        
    wb = to_px(white_box, w, h) if white_box else None
    balanced = white_balance(img, wb)
    
    strips = []
    annotated = img.copy()  # <--- Create the image we will draw on
    
    for n, box in enumerate(px_boxes, 1):
        rgb = pad_color(balanced, box)
        ph, dist = lab_to_ph(rgb_to_lab(rgb))
        
        conf = "high" if dist is not None and dist <= 8 else ("medium" if dist is not None and dist <= 15 else "low")
        if ph is None: conf = "none"
        
        status = "ok" if conf in ("high", "medium") else "needs_review"
        
        strips.append({
            "strip_id": f"strip_{n}", "ph": ph,
            "classification": classify(ph) if ph is not None else None,
            "confidence": conf, "color_distance": dist,
            "status": status,
            "box": [int(v) for v in box]
        })
        
        # --- DRAW ON THE IMAGE ---
        x1, y1, x2, y2 = box
        color = (0, 200, 0) if status == "ok" else (0, 165, 255) # Green if ok, Orange if needs review
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Add text label (pH value or "?")
        label = f"{n}: {ph:.2f}" if ph is not None else f"{n}: ?"
        fs2 = max(0.5, w / 900); t2 = max(1, int(round(fs2 * 2)))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs2, t2)
        
        # Draw black background for text
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, fs2, (0, 0, 0), t2)

    # Draw the white balance box if it exists
    if wb:
        cv2.rectangle(annotated, (wb[0], wb[1]), (wb[2], wb[3]), (255, 255, 255), 2)

    return {
        "id": rec_id, "num_strips": len(strips), "detector": detector,
        "chart": CHART_SOURCE, "strips": strips
    }, annotated

app = FastAPI(title="pH Reader")

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
        
    result, annotated_img = analyze_image(img, bx, wb, id)
    
    if "error" in result:
        return result
        
    # Encode the annotated image to base64
    ok, buf = cv2.imencode(".jpg", annotated_img)
    result["annotated_image_base64"] = base64.b64encode(buf).decode()
    
    return result
