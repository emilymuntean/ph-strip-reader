"""
One-time setup: tell the software where the pad windows and the white patch are on YOUR mat.

1. Glue the 4 marker tiles on the mat corners (make_markers.py).
2. Measure, in mm, the distance between marker CENTERS:  --width-mm (left<->right)
   and --height-mm (top<->bottom).
3. Take one clear top-down photo of the mat with all 4 markers visible.
4. Run:
      python define_layout.py mat_photo.jpg --width-mm 380 --height-mm 380
   A window shows the flattened mat. For each pad, drag a box over the WINDOW where the strip is
   wet (the zone that will be sampled - NOT the dry tail), press ENTER. Then drag a box over a
   plain WHITE area (white border of a marker tile or a glued white paper patch), press ENTER.
   Press C (or ESC) to skip/cancel a selection.
5. layout.json is written (plus layout_preview.jpg). Commit it next to server.py.

Pad names default to the 8-pad layout in the design history file; override with --names.
"""
import argparse
import json

import cv2
import numpy as np

import reader

DEFAULT_NAMES = [
    "L metatarsal", "L medial midfoot", "L lateral midfoot", "L heel",
    "R metatarsal", "R medial midfoot", "R lateral midfoot", "R heel",
]

ap = argparse.ArgumentParser()
ap.add_argument("photo")
ap.add_argument("--width-mm", type=float, required=True, help="marker-center to marker-center, left-right")
ap.add_argument("--height-mm", type=float, required=True, help="marker-center to marker-center, top-bottom")
ap.add_argument("--marker-mm", type=float, default=20.0)
ap.add_argument("--margin-mm", type=float, default=40.0, help="extra mat area to keep around the markers")
ap.add_argument("--names", default=",".join(DEFAULT_NAMES))
ap.add_argument("--out", default="layout.json")
a = ap.parse_args()

W, H = a.width_mm, a.height_mm
layout = {
    "marker_dict": "DICT_4X4_50",
    "marker_mm": a.marker_mm,
    "markers": {"0": [0, 0], "1": [W, 0], "2": [W, H], "3": [0, H]},
    "canvas_mm": [-a.margin_mm, -a.margin_mm, W + a.margin_mm, H + a.margin_mm],
    "pads": [],
    "white_box": [0, 0, 1, 1],
}

img = cv2.imread(a.photo)
corners, ids = reader.detect_markers(img, layout["marker_dict"])
Hm, seen, err = reader.fit_homography(corners, ids, layout)
if Hm is None:
    raise SystemExit(f"Need >= {reader.MIN_MARKERS} markers, found ids {seen}. Retake the photo.")
print(f"markers found: {seen}, alignment error {err:.2f} mm")

T, size = reader.canvas_transform(layout)
flat = cv2.warpPerspective(img, T @ Hm, size)
disp_scale = min(1.0, 1100.0 / max(flat.shape[:2]))
disp = cv2.resize(flat, None, fx=disp_scale, fy=disp_scale, interpolation=cv2.INTER_AREA)


def pick(title):
    x, y, w, h = cv2.selectROI(title, disp, showCrosshair=False)
    cv2.destroyAllWindows()
    if not (w and h):
        return None
    px = [x / disp_scale, y / disp_scale, (x + w) / disp_scale, (y + h) / disp_scale]
    x0, y0, _, _ = layout["canvas_mm"]
    S = reader.PX_PER_MM
    return [round(px[0] / S + x0, 1), round(px[1] / S + y0, 1),
            round(px[2] / S + x0, 1), round(px[3] / S + y0, 1)]


for name in [n.strip() for n in a.names.split(",") if n.strip()]:
    box = pick(f"PAD '{name}': drag over the wet window, ENTER (C to skip)")
    if box:
        layout["pads"].append({"name": name, "box": box})
wb = pick("WHITE patch: drag over plain white paper, ENTER")
if not wb:
    raise SystemExit("A white patch is required for white balance.")
layout["white_box"] = wb

with open(a.out, "w") as f:
    json.dump(layout, f, indent=2)

prev = flat.copy()
for p in layout["pads"]:
    l, t, r, b = reader.mm_box_to_px(p["box"], layout)
    cv2.rectangle(prev, (l, t), (r, b), (0, 200, 0), 2)
    cv2.putText(prev, p["name"], (l, max(15, t - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
l, t, r, b = reader.mm_box_to_px(layout["white_box"], layout)
cv2.rectangle(prev, (l, t), (r, b), (255, 0, 255), 2)
cv2.imwrite("layout_preview.jpg", prev)
print(f"Saved {a.out} with {len(layout['pads'])} pads. Check layout_preview.jpg.")
