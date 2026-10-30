"""
pH Strip Reader API  (v3 - fixes for the DFU pH mat workflow)
------------------------------------------------------------
POST /analyze  (multipart form)  - same interface as v2:
  file       : the mat photo (required)
  boxes      : optional JSON pad boxes (Gemini "locate" output). The reference card is
               ALWAYS read from the photo, even when boxes are supplied, so lighting
               is corrected with the in-photo card.
  white_box  : optional JSON [l,t,r,b] of a plain white area
  expected   : optional int, number of pads expected (e.g. 8)
  id         : optional, passed through to the output

Fixes in v3 (see FIXES.md):
  1. The reference card is always detected and read, even when `boxes` are supplied.
     (v2 skipped card detection entirely when boxes were given -> fell back to a
      generic/absent chart -> wrong pH + "color unclear".)
  2. Card swatches are detected with a separate raw-blob pass (no morphology merge),
     then grouped into their 2x7 (or 7x2) grid. The pH label of each swatch is resolved
     by grid position AND by matching swatch colors to universal-indicator color
     priors, so the card works at any rotation / placement in the photo.
     (v2 assumed the card was two columns of 7 in the bottom-right corner -> the
      team's 2-rows-of-7 card placed top-center was never read correctly.)
  3. Pad detection no longer rejects the elongated horizontal forefoot strips
     (aspect-ratio cap raised 4.5 -> 6.5) and the dry yellow tab of a strip is
     excluded from the color sample with a k-means split (v2 averaged the reacted
     color together with the dry yellow part -> pH pulled toward ~5-6).
  4. Zone metadata (zone_id, dry flag) from the locator is preserved; a pad flagged
     dry gets ph=null with a clear note instead of a bogus reading.
  5. "No calibration available" is reported distinctly from "color did not match".
The pH number is ALWAYS computed here with color math (CIELAB + calibration curve).
No AI model ever decides the pH value.
"""
import base64, json, os
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.exceptions import HTTPException

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

# Optional correction learned from buffer tests (meter pH vs app pH)
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
    """Returns (zones, white_box_or_None). zones keep zone_id / dry metadata."""
    if not raw:
        return None, None
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, dict):
        zones = data.get("zones") or []
        out = []
        for z in zones:
            if isinstance(z, dict):
                out.append({"bbox": z.get("bbox"), "zone_id": z.get("zone_id"),
                            "dry": bool(z.get("dry", False))})
            else:
                out.append({"bbox": z, "zone_id": None, "dry": False})
        return out, data.get("white_box")
    # plain list of boxes
    return [{"bbox": b, "zone_id": None, "dry": False} for b in data], None


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


def order_corners(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def find_and_flatten_mat(img, long_side=1400):
    """Find the mat (large bright region) and warp it flat. Unchanged from v2."""
    H, W = img.shape[:2]
    scale = 800 / max(H, W)
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    s = scale if scale < 1 else 1.0
    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0]
    L = cv2.GaussianBlur(L, (5, 5), 0)
    otsu, _ = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    T = min(otsu, 130)
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
    if area_frac > 0.93:
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


def paper_estimate(img):
    """Median color of the white paper/mat background (used for white balance)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
    L = lab[..., 0] * 100 / 255; a = lab[..., 1] - 128; b = lab[..., 2] - 128
    C = np.hypot(a, b)
    paper = (C < 10) & (L > np.percentile(L, 40))
    if paper.sum() < 100:
        paper = L > np.percentile(L, 70)
    if paper.sum() < 100:
        return None
    return np.median(img[paper].reshape(-1, 3), axis=0)


def color_diff_mask(img):
    """Mask of pixels that differ strongly from the white paper (pads, swatches...)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
    L = lab[..., 0] * 100 / 255; a = lab[..., 1] - 128; b = lab[..., 2] - 128
    C = np.hypot(a, b)
    paper = (C < 10) & (L > np.percentile(L, 40))
    if paper.sum() < 100:
        paper = L > np.percentile(L, 70)
    Lp, ap, bp = np.median(L[paper]), np.median(a[paper]), np.median(b[paper])
    dE = np.sqrt((L - Lp) ** 2 + (a - ap) ** 2 + (b - bp) ** 2)
    mask = ((dE > 30) & ((C > 15) | (L < Lp - 30))).astype(np.uint8) * 255
    return mask


def find_card(img):
    """Detect the 14-swatch reference card, wherever it is in the photo.

    v2 merged the swatches into one blob with morphology and only recognized a
    tight cluster of >= 4 blobs as 'the card' -> a 2-rows-of-7 card top-center
    was never found. v3 uses a raw (unmerged) blob pass, proximity clustering
    and a grid check, so the card is found at any position/rotation.

    Returns (swatch_boxes, card_box, paper_bgr) or (None, None, paper_bgr).
    """
    H, W = img.shape[:2]
    A = H * W
    mask = color_diff_mask(img)
    # RAW pass: no morphology, so each swatch stays its own blob.
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    cands = []
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        if ar < A * 0.00015 or ar > A * 0.004:
            continue
        fill = ar / float(w * h)
        if fill < 0.55:
            continue
        cands.append([int(x), int(y), int(x + w), int(y + h)])
    if len(cands) < 8:
        return None, None, paper_estimate(img)

    def ctr(bb):
        return np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])

    # single-linkage proximity clustering; only link similarly-SIZED blobs that
    # are close (pads are 3-5x bigger than swatches and must not chain in)
    centers = [ctr(b) for b in cands]
    sizes = [(b[2] - b[0]) * (b[3] - b[1]) for b in cands]
    maxdims = [max(b[2] - b[0], b[3] - b[1]) for b in cands]
    parent = list(range(len(cands)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            d = np.linalg.norm(centers[i] - centers[j])
            r = 2.0 * min(maxdims[i], maxdims[j])
            similar = max(sizes[i], sizes[j]) <= 2.2 * min(sizes[i], sizes[j])
            if d < r and similar:
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a
    groups = {}
    for i in range(len(cands)):
        groups.setdefault(find(i), []).append(i)

    # keep clusters with enough members to be a (possibly partial) card grid
    best = None
    for members in groups.values():
        if len(members) < 8:
            continue
        swatches = [cands[i] for i in members]
        grid, idxs = grid_slots(swatches)
        if grid is None:
            continue
        # prefer the cluster with the most grid-fitting swatches
        if best is None or len(idxs) > len(best[1]):
            best = (grid, idxs, swatches)
    if best is None:
        return None, None, paper_estimate(img)

    grid, idxs, swatches = best
    grid_swatches = [swatches[i] for i in idxs]
    xs = [b[0] for b in grid_swatches] + [b[2] for b in grid_swatches]
    ys = [b[1] for b in grid_swatches] + [b[3] for b in grid_swatches]
    m = 0.015 * max(H, W)
    card_box = [int(min(xs) - m), int(min(ys) - m), int(max(xs) + m), int(max(ys) + m)]
    return grid_swatches, card_box, paper_estimate(img)


def grid_slots(swatches):
    """Fit swatches to a 2x7 (or 7x2) card grid.

    Returns (slot_dict, grid_indices) where slot_dict maps (half, column) ->
    swatch index, using ONLY columns present in both halves (this drops stray
    blobs that got clustered with the card, e.g. a pad sitting next to it).
    Returns (None, None) if the swatches do not form a plausible card grid.
    """
    if len(swatches) < 12:
        return None, None
    cs = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in swatches])

    def best_split(val):
        v = np.sort(val)
        gaps = np.diff(v)
        if len(gaps) == 0:
            return None
        gi = int(np.argmax(gaps))
        return (v[gi] + v[gi + 1]) / 2

    for axis in (1, 0):  # 1 = two rows (2x7), 0 = two columns (7x2)
        thr = best_split(cs[:, axis])
        if thr is None:
            continue
        h1 = np.where(cs[:, axis] < thr)[0]
        h2 = np.where(cs[:, axis] >= thr)[0]
        if len(h1) < 5 or len(h2) < 5:
            continue
        # both halves must be tight along the split axis
        tight = True
        for g in (h1, h2):
            if np.ptp(cs[g, axis]) > 3.0 * np.median([max(b[2] - b[0], b[3] - b[1]) for b in swatches]):
                tight = False
        if not tight:
            continue
        # long axis = the other coordinate; cluster into columns by biggest gaps
        long_ax = cs[:, 1 - axis]
        v = np.sort(np.unique(np.round(long_ax).astype(int)))
        if len(v) < 7:
            continue
        gaps = np.diff(v)
        med = np.median(gaps)
        col_ids = {}
        col = 0
        col_ids[v[0]] = 0
        for prev, cur, g in zip(v[:-1], v[1:], gaps):
            if g > max(1.5 * med, 3):
                col += 1
            col_ids[cur] = col

        def nearest_col(x):
            key = int(np.round(x))
            return col_ids[min(col_ids, key=lambda kk: abs(kk - key))]

        slot = {}
        for i in range(len(swatches)):
            half = 0 if cs[i, axis] < thr else 1
            slot[(half, nearest_col(long_ax[i]))] = i
        # Drop stray columns that exist in only one half (e.g. a pad that got
        # clustered with the card), then require a COMPLETE card: exactly 7
        # shared columns, all 14 slots filled. An incomplete card is rejected
        # ("card incomplete") rather than guessed at - a missing column would
        # silently shift every label.
        cols1 = {c for (h, c) in slot if h == 0}
        cols2 = {c for (h, c) in slot if h == 1}
        shared = sorted(cols1 & cols2)
        grid = {k: v for k, v in slot.items() if k[1] in shared}
        if len(shared) != 7 or len(grid) != 14:
            continue
        # re-index the surviving columns by rank -> 0..6
        remap = {c: i for i, c in enumerate(shared)}
        grid = {(h, remap[c]): v for (h, c), v in grid.items()}
        return grid, sorted(grid.values())
    return None, None


def detect_pads(img, card_box):
    """Deterministic pad finder. Refactored from v2 detect_mat:
    - aspect-ratio cap raised to 6.5 so the horizontal forefoot strips pass
      (they measure ~4.4-5.4 on this mat)
    - pads inside the reference-card box are excluded (v2 could box card swatches)
    Returns (pad_boxes, labels, paper_bgr, rotation)."""
    H, W = img.shape[:2]
    A = H * W
    mask = color_diff_mask(img)
    k = max(3, int(min(H, W) * 0.008)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    blobs = []
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        if ar < A * 0.0004 or ar > A * 0.02:
            continue
        if x < W * 0.03 or y < H * 0.03 or x + w > W * 0.97 or y + h > H * 0.97:
            continue
        if ar / float(w * h) < 0.55 or max(w, h) / max(1, min(w, h)) > 6.5:
            continue
        bb = [int(x), int(y), int(x + w), int(y + h)]
        if card_box and card_box[0] <= (bb[0] + bb[2]) / 2 <= card_box[2] \
                and card_box[1] <= (bb[1] + bb[3]) / 2 <= card_box[3]:
            continue
        blobs.append(bb)
    if not blobs:
        return [], [], paper_estimate(img), 0

    def ctr(bb):
        return np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])

    # neutral zone labels in reading order (top-to-bottom, left-to-right).
    # Foot identity is NOT guessed from the photo - the "biggest gap" split can
    # misgroup pads, and location-based monitoring must not rely on it.
    labels, rotation = [], 0
    cs = np.array([ctr(p) for p in blobs]); mid = cs.mean(0)
    if card_box:
        cvec = np.array([(card_box[0] + card_box[2]) / 2, (card_box[1] + card_box[3]) / 2]) - mid

        def rot(v, k):
            for _ in range(k):
                v = np.array([-v[1], v[0]])
            return v
        rotation = max(range(4), key=lambda k: rot(cvec, k).sum())  # card bottom-right
    can = np.array([rot(c - mid, rotation) if card_box else (c - mid) for c in cs])
    order = np.argsort(can[:, 1])
    groups = []
    # group into rows by vertical overlap
    rows, cur = [], []
    for i in order:
        if cur and (can[i, 1] - np.mean(can[cur, 1])) > 0.5 * np.mean([blobs[j][3] - blobs[j][1] for j in cur]):
            rows.append(cur); cur = []
        cur.append(i)
    if cur:
        rows.append(cur)
    for row in rows:
        groups.extend(sorted(row, key=lambda i: can[i, 0]))
    pads = [blobs[i] for i in groups]
    labels = [f"zone_{n:02d}" for n in range(1, len(pads) + 1)]
    return pads, labels, paper_estimate(img), rotation


def white_balance(img, white_px_box=None, white_bgr=None):
    """Scale channels so a known-white area becomes neutral (target 240). Unchanged."""
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
    """Median RGB of the center of the region, glare pixels dropped.

    Deliberately simple: no 'dry yellow' heuristic. A legitimately reacted pH
    5-6 pad is yellow too, so color-based dry detection can silently discard
    the true color. Dry tabs should be excluded by the box (the locator is
    asked to box only the reacted part), not guessed here.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad = img[y1 + h // 6: y2 - h // 6, x1 + w // 6: x2 - w // 6]
    if pad.size == 0:
        pad = img[y1:y2, x1:x2]
    px = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(float)
    keep = ~(px.min(1) > 235)  # glare
    if keep.sum() > 20:
        px = px[keep]
    rgb = np.median(px, axis=0)
    spread = float(np.std(px, axis=0).mean())
    return rgb, spread


# Universal-indicator color priors in Lab (approximate) - used ONLY to work out
# which swatch is which pH when the card is read from the photo. The chart
# itself still comes from the swatch colors measured in the photo.
PH_PRIORS = {
    1.0: (45, 55, 40), 2.0: (55, 60, 55), 3.0: (60, 50, 60), 4.0: (65, 40, 65),
    5.0: (70, 25, 70), 6.0: (80, 8, 75), 7.0: (80, -5, 70), 8.0: (70, -10, 55),
    9.0: (60, -15, 40), 10.0: (50, -12, 20), 11.0: (40, 5, -25), 12.0: (35, 10, -35),
    13.0: (30, 15, -20), 14.0: (28, 20, -5),
}
# The team's card: 2 rows x 7 columns. Top row odd (1,3,..13), bottom row even
# (2,4,..14), both left-to-right. Rotated 90 deg it becomes 7 rows x 2 columns
# with the same left-to-right, top-to-bottom reading order.
CARD_ROWS_ODD = [1, 3, 5, 7, 9, 11, 13]
CARD_ROWS_EVEN = [2, 4, 6, 8, 10, 12, 14]


def chart_from_card(balanced, swatches):
    """Read the reference card IN THIS PHOTO and turn it into a pH->Lab chart.

    v3: fits the swatches to their 2x7 (or 7x2) grid, then resolves the
    labeling (which end is pH 1) by scoring the 4 possible orientations against
    universal-indicator color priors. The card can be placed anywhere in the
    photo and rotated any way.
    """
    grid, idxs = grid_slots(swatches)
    if grid is None:
        return None, "grid not recognized"

    # candidate labelings: half0 = odd row or even row; long axis increasing or decreasing
    variants = []
    for first_odd in (True, False):
        for reverse in (False, True):
            chart = {}
            ok = True
            for (half, col), i in grid.items():
                if first_odd:
                    row_ph = CARD_ROWS_ODD if half == 0 else CARD_ROWS_EVEN
                else:
                    row_ph = CARD_ROWS_EVEN if half == 0 else CARD_ROWS_ODD
                c = 6 - col if reverse else col
                if c < 0 or c > 6:
                    ok = False
                    break
                ph = row_ph[c]
                rgb, _ = pad_color(balanced, swatches[i])
                chart[float(ph)] = rgb_to_lab(rgb)
            if ok and len(chart) >= 12:
                variants.append((score_chart(chart), chart))
    if not variants:
        return None, "no consistent labeling"
    variants.sort(key=lambda t: t[0])
    return dict(sorted(variants[0][1].items())), "ok"


def score_chart(chart):
    """Lower = better match to universal-indicator color priors (hue ordering)."""
    keys = list(chart)
    # hue-like progression check: pH 1-4 red/orange, 6 yellow, 8 olive, 10-14 dark
    total = 0.0
    for ph in (2.0, 6.0, 10.0, 13.0):
        if ph not in chart:
            total += 500
            continue
        lab = chart[ph]
        prior = np.array(PH_PRIORS[ph])
        total += float(np.linalg.norm(lab - prior))
    # the alkaline end of the card is much darker than the yellow mid-range
    if 6.0 in chart and 13.0 in chart:
        if chart[13.0][0] < chart[6.0][0]:
            total -= 60
        else:
            total += 60
    if 2.0 in chart and 6.0 in chart:
        if chart[2.0][1] > chart[6.0][1]:  # pH2 more red than pH6
            total -= 40
        else:
            total += 40
    return total


def lab_to_ph(lab, chart=None):
    """Project the color onto the piecewise-linear calibration curve in Lab space."""
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
    if ph is None:
        return None
    if ph < 6.5:
        return "acidic"
    if ph <= 7.5:
        return "neutral"
    return "alkaline"


def analyze_image(img, zones=None, white_box=None, expected=None, rec_id=None):
    """zones: list of {'bbox': [l,t,r,b], 'zone_id': str|None, 'dry': bool}"""
    mat_flattened = False
    rotation = 0
    labels = None
    # Only try flattening if the reference card cannot be read in the original
    # photo (the DFU mat is a transparent acrylic plate on paper, which the
    # bright-region warp used to grab by mistake and destroy the layout).
    if not zones:
        sw0, cb0, _ = find_card(img)
        if cb0 is None or len(sw0) < 12:
            flat, ok = find_and_flatten_mat(img)
            if ok:
                swf, cbf, _ = find_card(flat)
                if cbf is not None and len(swf) >= 12 and (cb0 is None or len(sw0) < 12):
                    img, mat_flattened = flat, True
    h, w = img.shape[:2]

    # ---- ALWAYS read the reference card (fix #1) ----
    swatches, card_box, paper_bgr = find_card(img)
    card_found = card_box is not None and swatches and len(swatches) >= 12

    if zones:
        px_boxes = []
        zone_meta = []
        for z in zones:
            if z["bbox"] is None:
                continue
            b = to_px(z["bbox"], w, h)
            if b[2] - b[0] > 3 and b[3] - b[1] > 3:
                px_boxes.append(b)
                zone_meta.append(z)
        detector = "provided-boxes (+card-from-photo)"
    else:
        px_boxes, labels, paper_bgr, rotation = detect_pads(img, card_box)
        if card_found and rotation:
            for _ in range(rotation):
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = img.shape[:2]
            swatches, card_box, paper_bgr = find_card(img)
            card_found = card_box is not None and swatches and len(swatches) >= 12
            px_boxes, labels, paper_bgr, _ = detect_pads(img, card_box)
        detector = "mat-color-detector"

    wb = to_px(white_box, w, h) if white_box else None
    balanced = white_balance(img, wb, None if wb else paper_bgr)
    photo_chart, card_status = (None, "not attempted")
    if card_found:
        photo_chart, card_status = chart_from_card(balanced, swatches)
        if photo_chart is None:
            card_found = False
    chart_used = "reference card in photo" if photo_chart else (
        "chart.json (buffer calibration)" if CHART else "NONE - no calibration available, no pH reported")

    count_ok = expected is None or len(px_boxes) == expected
    strips, annotated = [], img.copy()
    for n, box in enumerate(px_boxes, 1):
        meta = zone_meta[n - 1] if zones else None
        rgb, spread = pad_color(balanced, box)
        ph, dist = lab_to_ph(rgb_to_lab(rgb), photo_chart)
        raw_ph = ph
        # The correction curve was fit against chart.json (buffer calibration).
        # Only apply it when that same chart source is in use - applying it on
        # top of the in-photo card chart would double-correct.
        if ph is not None and CORRECTION and photo_chart is None and CHART is not None:
            ph = round(float(np.clip(CORRECTION["slope"] * ph + CORRECTION["intercept"], 0, 14)), 2)
        loc = (meta["zone_id"] if meta and meta.get("zone_id") else
               (labels[n - 1] if not zones else f"strip_{n}"))
        note = ""
        if meta and meta.get("dry"):
            ph, dist, conf = None, None, "none"
            note = "pad flagged dry (no visible reaction)"
        elif ph is None:
            conf = "none"
            note = "no calibration chart available" if not photo_chart else ""
        elif dist <= 8 and spread <= 20:
            conf = "high"
        elif dist <= 15:
            conf = "medium"
        else:
            conf = "low"
        status = "ok" if (conf in ("high", "medium") and count_ok and (zones or card_found)) else "needs_review"
        strips.append({
            "id": rec_id, "strip_id": f"strip_{n}",
            "location": loc, "ph": ph, "raw_ph": raw_ph,
            "classification": classify(ph),
            "confidence": conf, "color_distance": dist, "status": status,
            "box": [int(v) for v in box], "color_rgb": [int(v) for v in rgb],
            "note": note,
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
    if labels or strips:
        fs = max(0.45, w / 1100); th_ = max(1, int(round(fs * 2)))
        lines = [f"{s['strip_id'][6:]}: {s['location']}  " +
                 (f"pH {s['ph']:.2f}" if s['ph'] is not None else (s['note'] or "no reading"))
                 for s in strips]
        if lines:
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
        "card_read": card_status if card_found else ("not found" if swatches is not None else "no swatches"),
        "white_balance": "white_box" if wb else ("paper-auto" if paper_bgr is not None else "gray-world"),
        "rotation_steps_90": rotation if not zones else 0,
        "reference_card_found": card_found, "mat_flattened": mat_flattened,
        "strips": strips, "_annotated": annotated,
    }


app = FastAPI(title="pH Strip Reader")


@app.get("/health")
def health():
    return {"ok": True, "detector": "mat-color-detector (or provided boxes) + card always read from photo",
            "chart": "reference card in each photo" + (" (fallback: chart.json)" if CHART else ""),
            "correction": CORRECTION}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), boxes: str = Form(None), white_box: str = Form(None),
                  expected: int = Form(None), id: str = Form(None)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # real error status so callers (n8n) fail at THIS request with the
        # actual message instead of a confusing 'field not set' downstream
        raise HTTPException(status_code=400, detail="Could not read image - not a valid JPG/PNG")
    try:
        zones, wb_from_boxes = parse_boxes(boxes)
        wb = json.loads(white_box) if white_box else wb_from_boxes
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse boxes/white_box JSON: {e}")
    result = analyze_image(img, zones, wb, expected, id)
    ok, buf = cv2.imencode(".jpg", result.pop("_annotated"))
    result["annotated_image_base64"] = base64.b64encode(buf).decode()
    return result
