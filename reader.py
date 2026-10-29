"""
reader.py - core pH reading logic for the DFU pH mat (v3).

What changed vs v2 and WHY (see FIXES.md):
  1. pH comes ONLY from chart.json (built from real buffers on YOUR strips). The printed
     1-14 universal card is no longer used for pH.
  2. Pads are located with ArUco markers on the mat + a fixed layout.json. No Gemini boxes.
  3. White balance uses a real white patch at a known place on the mat (not an LLM-chosen box),
     and is done in linear light so exposure differences cancel better.
  4. The color is sampled from a fixed window over the pad (the laser-cut window), not from a
     bounding box that may contain the dry yellow end of the strip, tape, or tissue.
  5. Every failure is reported with a specific reason instead of a generic "color unclear".
"""
import base64
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHART_PATH = os.environ.get("CHART_PATH", os.path.join(HERE, "chart.json"))
LAYOUT_PATH = os.environ.get("LAYOUT_PATH", os.path.join(HERE, "layout.json"))

# ---- tunables (calibrate.py prints suggested values for DIST_* after you build chart.json) ----
GAMMA = 2.2                 # sRGB approx; used to white-balance in linear light
WHITE_TARGET = 235.0        # sRGB value the white patch is scaled to (below clipping)
PX_PER_MM = 5.0             # resolution of the flattened ("canonical") mat image
PAD_SHRINK = 0.15           # ignore this fraction of the window on each side (edges/shadow)
DIST_HIGH = float(os.environ.get("DIST_HIGH", 6.0))     # Lab distance to chart curve: high conf
DIST_MED = float(os.environ.get("DIST_MED", 12.0))      # medium conf; above = low
SPREAD_MAX = float(os.environ.get("SPREAD_MAX", 8.0))   # mean pixel scatter inside a pad
GLARE_MAX_FRAC = 0.30
MIN_MARKERS = 3
MAX_ALIGN_ERR_MM = 2.0


# ------------------------------------------------------------------ loading
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_layout(path=None):
    lay = load_json(path or LAYOUT_PATH)
    if lay is None:
        raise FileNotFoundError(
            "layout.json not found. Run define_layout.py once on a photo of your mat.")
    return lay


def load_chart(path=None):
    raw = load_json(path or CHART_PATH)
    if not raw:
        return None
    return dict(sorted((float(k), np.array(v, float)) for k, v in raw.items()))


# ------------------------------------------------------------------ color helpers
def rgb01_to_lab(rgb01):
    """float RGB in 0..1, shape (...,3) -> Lab (L 0-100). No 8-bit rounding."""
    a = np.asarray(rgb01, np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)


def classify(ph):
    if ph < 6.5:
        return "acidic"
    if ph <= 7.5:
        return "neutral"
    return "alkaline"


# ------------------------------------------------------------------ markers / geometry
def detect_markers(img, dict_name="DICT_4X4_50"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(dic, params)
    corners, ids, _ = det.detectMarkers(gray)
    return corners, ids


def fit_homography(corners, ids, layout):
    """image px -> mat mm, from every corner of every known marker."""
    half = layout["marker_mm"] / 2.0
    src, dst, seen = [], [], []
    if ids is None:
        return None, [], None
    for c, i in zip(corners, ids.flatten()):
        key = str(int(i))
        if key not in layout["markers"] or int(i) in seen:
            continue
        cx, cy = layout["markers"][key]
        src.extend(c.reshape(4, 2))
        dst.extend([(cx - half, cy - half), (cx + half, cy - half),
                    (cx + half, cy + half), (cx - half, cy + half)])
        seen.append(int(i))
    if len(seen) < MIN_MARKERS:
        return None, seen, None
    src = np.float32(src)
    dst = np.float32(dst)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        return None, seen, None
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.linalg.norm(proj - dst, axis=1).mean())
    return H, seen, err


def canvas_transform(layout):
    x0, y0, x1, y1 = layout["canvas_mm"]
    S = PX_PER_MM
    T = np.array([[S, 0, -x0 * S], [0, S, -y0 * S], [0, 0, 1]], float)
    size = (int(round((x1 - x0) * S)), int(round((y1 - y0) * S)))
    return T, size


def mm_box_to_px(box, layout):
    x0, y0, _, _ = layout["canvas_mm"]
    S = PX_PER_MM
    l, t, r, b = box
    return [int(round((l - x0) * S)), int(round((t - y0) * S)),
            int(round((r - x0) * S)), int(round((b - y0) * S))]


# ------------------------------------------------------------------ white balance
def white_balance(flat_bgr, white_px):
    """Scale channels in LINEAR light so the white patch becomes neutral at WHITE_TARGET."""
    rgb = flat_bgr[:, :, ::-1].astype(np.float32) / 255.0
    lin = rgb ** GAMMA
    x1, y1, x2, y2 = white_px
    patch = lin[y1:y2, x1:x2].reshape(-1, 3)
    if patch.size == 0:
        raise ValueError("white_box lies outside the flattened mat")
    ref = np.median(patch, axis=0)
    target = (WHITE_TARGET / 255.0) ** GAMMA
    gains = target / np.maximum(ref, 1e-4)
    out = np.clip(lin * gains, 0, 1) ** (1.0 / GAMMA)
    white_srgb = (ref ** (1.0 / GAMMA)) * 255.0
    patch_spread = float(np.std(patch ** (1.0 / GAMMA) * 255.0, axis=0).mean())
    return out, gains, white_srgb, patch_spread


# ------------------------------------------------------------------ pad sampling
def sample_pad(bal01, box_px):
    x1, y1, x2, y2 = box_px
    dx = int((x2 - x1) * PAD_SHRINK)
    dy = int((y2 - y1) * PAD_SHRINK)
    roi = bal01[y1 + dy:y2 - dy, x1 + dx:x2 - dx].reshape(-1, 3)
    if roi.shape[0] < 20:
        return None
    lab = rgb01_to_lab(roi)
    C = np.hypot(lab[:, 1], lab[:, 2])
    glare = (lab[:, 0] > 93) & (C < 10)
    glare_frac = float(glare.mean())
    if (~glare).sum() >= 20:
        lab = lab[~glare]
        roi = roi[~glare]
    med = np.median(lab, axis=0)
    spread = float(np.mean(np.linalg.norm(lab - med, axis=1)))
    rgb = (np.median(roi, axis=0) * 255).round().astype(int).tolist()
    return {"lab": med, "rgb": rgb, "spread": spread, "glare_frac": glare_frac}


def measure(img, layout=None):
    """Find markers, flatten, white-balance and read every pad's Lab color. No pH here."""
    layout = layout or load_layout()
    out = {"markers_found": False, "markers_seen": [], "align_err_mm": None,
           "warnings": [], "pads": [], "H": None}
    corners, ids = detect_markers(img, layout.get("marker_dict", "DICT_4X4_50"))
    out["_corners"], out["_ids"] = corners, ids
    H, seen, err = fit_homography(corners, ids, layout)
    out["markers_seen"], out["align_err_mm"] = seen, err
    if H is None:
        out["warnings"].append(
            f"only {len(seen)} of 4 alignment markers visible - retake photo with all 4 corners in view")
        return out
    out["markers_found"] = True
    out["H"] = H
    if err > MAX_ALIGN_ERR_MM:
        out["warnings"].append(
            f"marker alignment error {err:.1f} mm - mat may be bent, a marker is damaged, or the photo is blurred")

    T, size = canvas_transform(layout)
    flat = cv2.warpPerspective(img, T @ H, size, flags=cv2.INTER_LINEAR)
    out["_flat"] = flat

    wpx = mm_box_to_px(layout["white_box"], layout)
    bal, gains, white_srgb, wspread = white_balance(flat, wpx)
    out["white_srgb"] = [round(float(v), 1) for v in white_srgb]
    out["wb_gains"] = [round(float(v), 3) for v in gains]
    if white_srgb.max() > 250:
        out["warnings"].append("white reference is overexposed - reduce glare/brightness and retake")
    if white_srgb.min() < 80:
        out["warnings"].append("white reference is too dark - add light and retake")
    if wspread > 12:
        out["warnings"].append("white reference is uneven (shadow, glare or tape on it) - retake")
    if gains.max() / gains.min() > 2.2:
        out["warnings"].append("strong color cast in the lighting - use white/daylight light and retake")

    for pad in layout["pads"]:
        bpx = mm_box_to_px(pad["box"], layout)
        s = sample_pad(bal, bpx)
        rec = {"name": pad["name"], "box_mm": pad["box"], "box_px": bpx}
        if s:
            rec.update(s)
        out["pads"].append(rec)
    out["_white_px"] = wpx
    return out


# ------------------------------------------------------------------ lab -> pH
def lab_to_ph(lab, chart):
    """Project onto the piecewise-linear buffer curve in Lab space.
    Returns (ph, distance_to_curve, range_flag)."""
    keys = list(chart)
    best = (1e9, keys[0], 0, 0)
    for i in range(len(keys) - 1):
        p0, p1 = chart[keys[i]], chart[keys[i + 1]]
        seg = p1 - p0
        t = float(np.clip(np.dot(lab - p0, seg) / max(np.dot(seg, seg), 1e-9), 0, 1))
        d = float(np.linalg.norm(lab - (p0 + t * seg)))
        if d < best[0]:
            best = (d, keys[i] + t * (keys[i + 1] - keys[i]), i, t)
    d, ph, i, t = best
    flag = None
    if i == 0 and t <= 0.02:
        flag = "below"
    elif i == len(keys) - 2 and t >= 0.98:
        flag = "above"
    return round(ph, 2), round(d, 1), flag


# ------------------------------------------------------------------ full analysis
def analyze(img, rec_id=None, layout=None, chart=None):
    layout = layout or load_layout()
    chart = chart if chart is not None else load_chart()
    m = measure(img, layout)
    warnings = list(m["warnings"])
    strips = []
    blocking = not m["markers_found"]

    if chart is None:
        warnings.append("chart.json missing on the server - no pH can be reported")
    chart_range = (min(chart), max(chart)) if chart else (None, None)

    for n, p in enumerate(m["pads"], 1):
        rec = {"id": rec_id, "strip_id": f"strip_{n}", "location": p["name"], "ph": None,
               "raw_ph": None, "classification": None, "confidence": "none",
               "color_distance": None, "range_flag": None, "status": "needs_review",
               "box": p["box_px"], "color_rgb": p.get("rgb"), "note": ""}
        notes = []
        if blocking:
            notes.append("markers not found")
        elif "lab" not in p:
            notes.append("pad window not readable")
        elif chart is None:
            notes.append("no calibration chart")
        else:
            ph, dist, flag = lab_to_ph(p["lab"], chart)
            rec["raw_ph"], rec["color_distance"], rec["range_flag"] = ph, dist, flag
            if dist <= DIST_HIGH and p["spread"] <= SPREAD_MAX:
                conf = "high"
            elif dist <= DIST_MED and p["spread"] <= SPREAD_MAX * 1.5:
                conf = "medium"
            else:
                conf = "low"
            rec["confidence"] = conf
            if conf == "low":
                if p["spread"] > SPREAD_MAX * 1.5:
                    notes.append("uneven color across pad (partly wet, bleeding or tape)")
                else:
                    notes.append("color does not match calibration (dry/missing strip, wrong strip type, or lighting)")
            if p["glare_frac"] > GLARE_MAX_FRAC:
                notes.append("glare on pad")
            if flag == "below":
                notes.append(f"below strip range (<= {chart_range[0]:g})")
            elif flag == "above":
                notes.append(f"above strip range (>= {chart_range[1]:g})")
            if conf in ("high", "medium") and p["glare_frac"] <= GLARE_MAX_FRAC:
                rec["ph"] = ph
                rec["classification"] = classify(ph)
                rec["status"] = "ok" if not warnings_block(m) else "needs_review"
        rec["note"] = "; ".join(notes)
        strips.append(rec)

    return {
        "id": rec_id,
        "num_strips": len(strips),
        "expected": len(layout["pads"]),
        "count_ok": True,
        "detector": "aruco-layout",
        "chart": "chart.json (buffer calibration)" if chart else "NONE",
        "markers_found": m["markers_found"],
        "reference_card_found": m["markers_found"],  # kept so old n8n nodes don't break
        "markers_seen": m["markers_seen"],
        "align_err_mm": None if m["align_err_mm"] is None else round(m["align_err_mm"], 2),
        "white_srgb": m.get("white_srgb"),
        "wb_gains": m.get("wb_gains"),
        "warnings": warnings,
        "strips": strips,
        "_annotated": annotate(img, m, strips),
        "_flat": m.get("_flat"),
    }


def warnings_block(m):
    """Warnings that make every pad untrustworthy (lighting / alignment problems)."""
    return [w for w in m["warnings"]
            if any(k in w for k in ("white reference", "color cast", "alignment error"))]


# ------------------------------------------------------------------ drawing
def annotate(img, m, strips):
    out = img.copy()
    h, w = out.shape[:2]
    th = max(2, w // 600)
    fs = max(0.5, w / 1600)
    if m.get("_ids") is not None and len(m["_ids"]):
        cv2.aruco.drawDetectedMarkers(out, m["_corners"], m["_ids"])
    return _finish_annotation(out, m, strips, th, fs)


def _finish_annotation(out, m, strips, th, fs):
    if m["H"] is not None:
        Hinv = np.linalg.inv(m["H"])
        for n, (s, p) in enumerate(zip(strips, m["pads"]), 1):
            l, t, r, b = p["box_mm"]
            quad = np.float32([[l, t], [r, t], [r, b], [l, b]]).reshape(-1, 1, 2)
            pts = cv2.perspectiveTransform(quad, Hinv).reshape(-1, 2).astype(int)
            col = (0, 200, 0) if s["status"] == "ok" else (0, 165, 255)
            cv2.polylines(out, [pts], True, col, th)
            txt = f"{n}" + (f" {s['ph']:.2f}" if s["ph"] is not None else " ?")
            cv2.putText(out, txt, tuple(pts[0] + [0, -6]), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 0, 0), th + 2)
            cv2.putText(out, txt, tuple(pts[0] + [0, -6]), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        col, th)
    scale = 1600.0 / max(out.shape[:2])
    if scale < 1:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out


def encode_b64(img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


if __name__ == "__main__":
    import sys
    im = cv2.imread(sys.argv[1])
    res = analyze(im, rec_id=os.path.basename(sys.argv[1]))
    res.pop("_annotated"); res.pop("_flat")
    print(json.dumps(res, indent=2, default=float))
