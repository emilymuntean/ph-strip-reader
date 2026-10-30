import base64
import io
import json
import os

import cv2
import numpy as np

GAMMA = 2.2
WHITE_TARGET = 235.0
PAD_SHRINK = 0.15
DIST_HIGH = float(os.environ.get("DIST_HIGH", 8.0))
DIST_MED = float(os.environ.get("DIST_MED", 15.0))
GLARE_L_MAX = 93.0
GLARE_C_MAX = 10.0


def rgb01_to_lab(rgb01):
    a = np.asarray(rgb01, np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)


def classify(ph):
    if ph < 6.5: return "acidic"
    if ph <= 7.5: return "neutral"
    return "alkaline"


def white_balance_with_box(img_bgr, box_norm):
    h, w = img_bgr.shape[:2]
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    lin = rgb ** GAMMA
    
    try:
        l, t, r, b = box_norm
        x1 = max(0, int(l * w)); y1 = max(0, int(t * h))
        x2 = min(w, int(r * w)); y2 = min(h, int(b * h))
        if x2 - x1 < 5 or y2 - y1 < 5:
            raise ValueError("Box too small")
        patch = lin[y1:y2, x1:x2].reshape(-1, 3)
        ref = np.median(patch, axis=0)
    except Exception:
        # Fallback to brightest 1%
        flat_lin = lin.reshape(-1, 3)
        brightest_idx = np.argsort(flat_lin.sum(axis=1))[-int(len(flat_lin)*0.01):]
        ref = np.median(flat_lin[brightest_idx], axis=0)

    target = (WHITE_TARGET / 255.0) ** GAMMA
    gains = target / np.maximum(ref, 1e-4)
    out = np.clip(lin * gains, 0, 1) ** (1.0 / GAMMA)
    
    white_srgb = (ref ** (1.0 / GAMMA)) * 255.0
    return out, {"white_srgb": [round(float(v), 1) for v in white_srgb], "gains": [round(float(v), 3) for v in gains]}


def sample_box_lab(bal01, box_norm, shrink=0.15):
    h, w = bal01.shape[:2]
    l, t, r, b = box_norm
    x1 = max(0, int(l * w)); y1 = max(0, int(t * h))
    x2 = min(w, int(r * w)); y2 = min(h, int(b * h))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None, None

    # Shrink the box to avoid edges, glare, and the plastic border
    dx = int((x2 - x1) * shrink)
    dy = int((y2 - y1) * shrink)
    roi = bal01[y1 + dy:y2 - dy, x1 + dx:x2 - dx].reshape(-1, 3)
    if roi.shape[0] < 20:
        return None, None

    lab = rgb01_to_lab(roi)
    C = np.hypot(lab[:, 1], lab[:, 2])
    glare = (lab[:, 0] > GLARE_L_MAX) & (C < GLARE_C_MAX)
    
    if (~glare).sum() >= 20:
        lab = lab[~glare]
        roi = roi[~glare]
        
    med = np.median(lab, axis=0)
    return med, {"spread": float(np.mean(np.linalg.norm(lab - med, axis=1))), "glare_frac": float(glare.mean())}


def build_chart_from_card(bal01, swatches):
    chart = {}
    for sw in swatches:
        try: 
            ph = float(sw["ph"])
            bbox = sw["bbox"]
        except: 
            continue
        
        # Shrink by 25% for swatches to get the purest center color
        lab, _ = sample_box_lab(bal01, bbox, shrink=0.25)
        
        if lab is not None: 
            # Reject swatches that are actually just white/glare (L > 90 and low saturation)
            L, a, b = lab
            C = np.hypot(a, b)
            if L > 88 and C < 12:
                continue # This is a glare spot, not a real swatch
                
            chart[ph] = lab
            
    return dict(sorted(chart.items()))


def lab_to_ph(lab, chart):
    keys = list(chart)
    best = (1e9, keys[0], 0, 0.0)
    for i in range(len(keys) - 1):
        p0, p1 = chart[keys[i]], chart[keys[i + 1]]
        seg = p1 - p0
        t = float(np.clip(np.dot(lab - p0, seg) / max(float(np.dot(seg, seg)), 1e-9), 0, 1))
        d = float(np.linalg.norm(lab - (p0 + t * seg)))
        if d < best[0]: 
            best = (d, keys[i] + t * (keys[i + 1] - keys[i]), i, t)
            
    d, ph, i, t = best
    flag = "below" if (i == 0 and t <= 0.02) else ("above" if (i == len(keys) - 2 and t >= 0.98) else None)
    return round(float(ph), 2), round(float(d), 1), flag


def analyze_with_boxes(img_bgr, loc, rec_id=None):
    if loc.get("needs_retake"):
        return {"id": rec_id, "error": "Retake photo: " + (loc.get("reason") or "unreadable"), "needs_retake": True, "strips": [], "swatches_found": 0}

    strips_in = loc.get("strips") or []
    swatches_in = loc.get("swatches") or []
    white_box = loc.get("white_box")

    try:
        wb, wb_info = white_balance_with_box(img_bgr, white_box)
    except Exception as e:
        return {"id": rec_id, "error": f"White balance failed: {e}", "needs_retake": True, "strips": [], "swatches_found": 0}

    chart = build_chart_from_card(wb, swatches_in)
    
    if len(chart) < 2:
        return {"id": rec_id, "error": f"Only {len(chart)} swatches readable (need >=2)", "needs_retake": True, "strips": [], "swatches_found": len(chart)}

    chart_range = (min(chart), max(chart))
    strips_out = []
    
    strips_in = strips_in[:8]
    
    for i, s in enumerate(strips_in, 1):
        rec = {"strip_id": s.get("zone_id") or f"strip_{i}", "ph": None, "classification": None, "confidence": "none", "color_distance": None, "range_flag": None, "status": "needs_review", "note": ""}
        lab, meta = sample_box_lab(wb, s.get("bbox") or [0,0,0,0], shrink=0.15)
        
        if lab is None:
            rec["note"] = "strip box unreadable"
            strips_out.append(rec)
            continue

        ph, dist, flag = lab_to_ph(lab, chart)
        rec["ph"], rec["classification"], rec["color_distance"], rec["range_flag"] = ph, classify(ph), dist, flag
        rec["confidence"] = "high" if dist <= DIST_HIGH else ("medium" if dist <= DIST_MED else "low")
        
        notes = []
        if meta and meta.get("spread", 0) > 12: notes.append("uneven color")
        if meta and meta.get("glare_frac", 0) > 0.30: notes.append("glare")
        
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
        "strips": strips_out, 
        "_wb_image": wb
    }


def annotate(img_bgr, loc):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    th = max(2, w // 600)
    
    def draw(box, color, label):
        l, t, r, b = box
        cv2.rectangle(out, (int(l*w), int(t*h)), (int(r*w), int(b*h)), color, th)
        if label: cv2.putText(out, label, (int(l*w), max(15, int(t*h)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, th)
        
    for s in loc.get("strips") or []: draw(s.get("bbox"), (0, 200, 0), s.get("zone_id", ""))
    for sw in loc.get("swatches") or []: draw(sw.get("bbox"), (255, 0, 0), f"pH {sw.get('ph','?')}")
    if loc.get("white_box"): draw(loc["white_box"], (255, 0, 255), "WB")
    return out


def encode_b64(img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()
