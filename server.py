"""
pH Strip Reader API  (v2 - for the DFU pH mat workflow)
-------------------------------------------------------
POST /analyze  (multipart form)
  file       : the mat photo (required)
  (default)  : no boxes needed. The mat detector finds the pads by color, ignores tape/lines,
               excludes the reference card, and uses the card (bottom-right corner) to name
               each pad by location no matter how the phone was rotated.
  boxes      : optional JSON (only if you want to supply your own pad boxes). Pad locations, normalized 0-1, [left, top, right, bottom].
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
    return None, "none"


CHART, CHART_SOURCE = load_chart()

# Optional correction learned from buffer tests (meter pH vs app pH), see fit_correction.py
CORRECTION_PATH = os.path.join(HERE, "correction.json")
CORRECTION = None
if os.path.exists(CORRECTION_PATH):
    with open(CORRECTION_PATH) as f:
        CORRECTION = json.load(f)  # {"slope": a, "intercept": b}

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


def order_corners(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def find_and_flatten_mat(img, long_side=1400):
    """Find the mat (large bright region) and warp it flat, so photos taken at an angle,
    from far away, or with background in view all look like a straight top-down photo.
    Returns (flat_image, found_bool)."""
    H, W = img.shape[:2]
    scale = 800 / max(H, W)
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    s = scale if scale < 1 else 1.0
    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0]
    L = cv2.GaussianBlur(L, (5, 5), 0)
    otsu, _ = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    T = min(otsu, 130)  # keep the gray tape border as part of the mat, drop the dark floor
    m = (L > T).astype(np.uint8) * 255
    k = max(5, int(max(small.shape[:2]) * 0.02)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, False
    c = max(cnts, key=cv2.contourArea)
    area_frac = cv2.contourArea(c) / float(small.shape[0] * small.shape[1])
    if area_frac < 0.03:
        return img, False
    if area_frac > 0.93:            # mat already fills the photo
        return img, False
    hull = cv2.convexHull(c)
    quad = None
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.08, 15):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2); break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))
    q = order_corners(quad / s)
    wA = np.linalg.norm(q[1] - q[0]); wB = np.linalg.norm(q[2] - q[3])
    hA = np.linalg.norm(q[3] - q[0]); hB = np.linalg.norm(q[2] - q[1])
    w, h = max(wA, wB), max(hA, hB)
    f = long_side / max(w, h)
    w, h = int(w * f), int(h * f)
    M = cv2.getPerspectiveTransform(q, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32))
    return cv2.warpPerspective(img, M, (w, h)), True


def detect_mat(img):
    """Deterministic pad finder for the DFU mat (no AI).
    1. Estimate the white paper color.
    2. Find compact blobs that differ strongly from paper (pads + reference card swatches).
       Pale tape, pencil lines, gray border and shadows are rejected by color/size/shape.
    3. Reference card = tight cluster of many swatches -> excluded from pads.
    4. Orientation: the card marks the bottom-right corner, so pads are named the same way
       no matter how the phone was rotated."""
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
    L = lab[..., 0] * 100 / 255; a = lab[..., 1] - 128; b = lab[..., 2] - 128
    C = np.hypot(a, b)
    paper = (C < 10) & (L > np.percentile(L, 40))
    if paper.sum() < 100:
        paper = L > np.percentile(L, 70)
    Lp, ap, bp = np.median(L[paper]), np.median(a[paper]), np.median(b[paper])
    paper_bgr = np.median(img[paper].reshape(-1, 3), axis=0)
    dE = np.sqrt((L - Lp) ** 2 + (a - ap) ** 2 + (b - bp) ** 2)
    mask = ((dE > 35) & ((C > 15) | (L < Lp - 35))).astype(np.uint8) * 255
    k = max(3, int(min(H, W) * 0.008)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    A = H * W
    blobs = []
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        if ar < A * 0.0004 or ar > A * 0.02:
            continue
        if x < W * 0.03 or y < H * 0.03 or x + w > W * 0.97 or y + h > H * 0.97:
            continue  # border / shadows
        if ar / float(w * h) < 0.6 or max(w, h) / max(1, min(w, h)) > 4.5:
            continue  # not a compact rectangle (lines, outlines)
        blobs.append([int(x), int(y), int(x + w), int(y + h)])

    def ctr(bb):
        return np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
    # reference card = blobs with >=3 close neighbours (swatch grid)
    card_idx = set()
    for i, bi in enumerate(blobs):
        r = 2.5 * max(bi[2] - bi[0], bi[3] - bi[1])
        nb = sum(1 for j, bj in enumerate(blobs) if j != i and np.linalg.norm(ctr(bi) - ctr(bj)) < r)
        if nb >= 3:
            card_idx.add(i)
    card_box = None
    if len(card_idx) >= 4:
        cb = np.array([blobs[i] for i in card_idx])
        m = 0.02 * max(H, W)
        card_box = [int(cb[:, 0].min() - m), int(cb[:, 1].min() - m),
                    int(cb[:, 2].max() + m), int(cb[:, 3].max() + m)]
    pads = []
    for i, bb in enumerate(blobs):
        if i in card_idx:
            continue
        c = ctr(bb)
        if card_box and card_box[0] <= c[0] <= card_box[2] and card_box[1] <= c[1] <= card_box[3]:
            continue
        pads.append(bb)

    # ---- orientation + location names ----
    labels, rotation = [], None
    if pads:
        def rot(v, k):
            for _ in range(k):
                v = np.array([-v[1], v[0]])  # 90 degree step
            return v
        cs = np.array([ctr(p) for p in pads]); mid = cs.mean(0)
        if card_box:
            cvec = np.array([(card_box[0] + card_box[2]) / 2, (card_box[1] + card_box[3]) / 2]) - mid
            rotation = max(range(4), key=lambda k: rot(cvec, k).sum())  # put card bottom-right
        else:
            rotation = 0
        can = np.array([rot(c - mid, rotation) for c in cs])
        order = np.argsort(can[:, 1])
        if len(pads) >= 2:
            g = int(np.argmax(np.diff(can[order, 1]))) + 1  # biggest vertical gap = between feet
            groups = [list(order[:g]), list(order[g:])]
            foot_names = ["Top foot", "Bottom foot"]
        else:
            groups, foot_names = [list(order)], ["Mat"]
        pos = {1: ["center"], 2: ["left", "right"], 3: ["left", "middle", "right"]}
        lab_of = {}
        for gname, grp in zip(foot_names, groups):
            grp = sorted(grp, key=lambda i: can[i, 0])
            names = pos.get(len(grp), [f"#{j + 1}" for j in range(len(grp))])
            for i, nm in zip(grp, names):
                lab_of[i] = f"{gname} - {nm}"
        idx = sorted(range(len(pads)), key=lambda i: (lab_of[i].startswith("Bottom"), can[i, 0]))
        pads = [pads[i] for i in idx]; labels = [lab_of[i] for i in idx]
    swatches = [blobs[i] for i in sorted(card_idx)]
    return pads, labels, card_box, paper_bgr, rotation, swatches


def white_balance(img, white_px_box=None, white_bgr=None):
    """Scale channels so a known-white area becomes neutral (target 240). Falls back to gray-world."""
    f = img.astype(np.float32)
    if white_bgr is not None:
        ref = np.asarray(white_bgr, np.float32)
        target = 240.0
    elif white_px_box:
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



# Universal indicator card layout (canonical orientation, card upright in bottom-right):
# left column top->bottom = 13,11,9,7,5,3,1 ; right column top->bottom = 14,12,10,8,6,4,2
CARD_LEFT = [13, 11, 9, 7, 5, 3, 1]
CARD_RIGHT = [14, 12, 10, 8, 6, 4, 2]


def chart_from_card(balanced, swatches, rotation):
    """Read the reference card IN THIS PHOTO and turn it into a pH->Lab chart.
    Because the card is under the same light as the pads, lighting differences cancel out."""
    if len(swatches) != 14:
        return None
    def rot(v, k):
        for _ in range(k or 0):
            v = np.array([-v[1], v[0]])
        return v
    cs = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in swatches])
    mid = cs.mean(0)
    can = np.array([rot(c - mid, rotation) for c in cs])
    order = np.argsort(can[:, 0])
    left, right = sorted(order[:7], key=lambda i: can[i, 1]), sorted(order[7:], key=lambda i: can[i, 1])
    chart = {}
    for col, phs in ((left, CARD_LEFT), (right, CARD_RIGHT)):
        for i, ph in zip(col, phs):
            rgb, _ = pad_color(balanced, swatches[i])
            chart[float(ph)] = rgb_to_lab(rgb)
    return dict(sorted(chart.items()))

def lab_to_ph(lab, chart=None):
    """Project the color onto the piecewise-linear calibration curve in Lab space.
    Gives a continuous pH between chart points (needed for +/-0.2 precision)."""
    chart = chart or CHART
    if not chart:
        return None, None
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
    if ph < 6.5:
        return "acidic"
    if ph <= 7.5:
        return "neutral"
    return "alkaline"


def analyze_image(img, boxes=None, white_box=None, expected=None, rec_id=None):
    mat_flattened = False
    if not boxes:
        flat, ok = find_and_flatten_mat(img)
        if ok:
            # use the flattened mat only if it reads at least as well as the original photo
            def score(im):
                p, _, cb, _, _, sw = detect_mat(im)
                return (cb is not None and len(sw) == 14, expected is None or len(p) == expected, len(p))
            if score(flat) >= score(img):
                img, mat_flattened = flat, True
    h, w = img.shape[:2]
    labels, card_box, paper_bgr, rotation, swatches = None, None, None, None, []
    if boxes:
        px_boxes, detector = [to_px(b, w, h) for b in boxes], "provided-boxes"
        px_boxes = sort_reading_order([b for b in px_boxes if b[2] - b[0] > 3 and b[3] - b[1] > 3])
    else:
        px_boxes, labels, card_box, paper_bgr, rotation, swatches = detect_mat(img)
        if card_box is not None and rotation:
            # turn the photo upright (card bottom-right) so the annotated image is easy to read
            for _ in range(rotation):
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            px_boxes, labels, card_box, paper_bgr, rotation, swatches = detect_mat(img)
            h, w = img.shape[:2]
        detector = "mat-color-detector"

    wb = to_px(white_box, w, h) if white_box else None
    balanced = white_balance(img, wb, None if wb else paper_bgr)
    photo_chart = chart_from_card(balanced, swatches, rotation) if swatches else None
    chart_used = "reference card in photo" if photo_chart else (
        "chart.json (buffer calibration)" if CHART else "NONE - reference card not readable, no pH reported")

    count_ok = expected is None or len(px_boxes) == expected
    orientation_ok = bool(boxes) or card_box is not None
    strips, annotated = [], img.copy()
    for n, box in enumerate(px_boxes, 1):
        rgb, spread = pad_color(balanced, box)
        ph, dist = lab_to_ph(rgb_to_lab(rgb), photo_chart)
        raw_ph = ph
        if ph is not None and CORRECTION:
            ph = round(float(np.clip(CORRECTION["slope"] * ph + CORRECTION["intercept"], 0, 14)), 2)
        if ph is None:
            conf = "none"
        elif dist <= 8 and spread <= 20:
            conf = "high"
        elif dist <= 15:
            conf = "medium"
        else:
            conf = "low"
        status = "ok" if (conf in ("high", "medium") and count_ok and orientation_ok) else "needs_review"
        strips.append({
            "id": rec_id, "strip_id": f"strip_{n}",
            "location": labels[n - 1] if labels else f"strip_{n}", "ph": ph, "raw_ph": raw_ph,
            "classification": classify(ph) if ph is not None else None,
            "confidence": conf, "color_distance": dist, "status": status,
            "box": [int(v) for v in box], "color_rgb": [int(v) for v in rgb],
        })
        x1, y1, x2, y2 = box
        color = (0, 200, 0) if status == "ok" else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = str(n)
        fs2 = max(0.5, w / 900); t2 = max(1, int(round(fs2 * 2)))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs2, t2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, fs2, (0, 0, 0), t2)
    if wb:
        cv2.rectangle(annotated, (wb[0], wb[1]), (wb[2], wb[3]), (255, 255, 255), 2)
    if card_box:
        cv2.rectangle(annotated, tuple(card_box[:2]), tuple(card_box[2:]), (255, 0, 255), 2)
        cv2.putText(annotated, "ref card", (card_box[0], max(12, card_box[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    # legend with location names (dark box, readable at any photo size)
    if labels:
        fs = max(0.45, w / 1100); th_ = max(1, int(round(fs * 2)))
        lines = [f"{s['strip_id'][6:]}: {s['location']}  " + (f"pH {s['ph']:.2f}" if s['ph'] is not None else "no reading") for s in strips]
        sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th_)[0] for t in lines]
        lh = max(sz[1] for sz in sizes) + int(10 * fs + 6)
        bw = max(sz[0] for sz in sizes) + 16
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (bw, lh * len(lines) + 8), (0, 0, 0), -1)
        annotated[:] = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
        for i, t in enumerate(lines):
            cv2.putText(annotated, t, (8, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th_)
    return {
        "id": rec_id, "num_strips": len(strips), "expected": expected, "count_ok": count_ok,
        "detector": detector, "chart": chart_used,
        "white_balance": "white_box" if wb else ("paper-auto" if paper_bgr is not None else "gray-world"),
        "rotation_steps_90": rotation, "reference_card_found": card_box is not None,
        "orientation_ok": orientation_ok, "mat_flattened": mat_flattened,
        "strips": strips, "_annotated": annotated,
    }


app = FastAPI(title="pH Strip Reader")


@app.get("/health")
def health():
    return {"ok": True, "detector": "mat-color-detector (or provided boxes)",
            "chart": "reference card in each photo" + (" (fallback: chart.json)" if CHART else ""),
            "correction": CORRECTION}


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
