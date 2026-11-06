"""
Build chart.json for YOUR strips  (run on your laptop, then upload chart.json to the Hugging Face Space)
... (keep your original docstring) ...
"""
import json, os, sys
import cv2
import numpy as np
from server import white_balance, rgb_to_lab, pad_color, CHART_PATH

MAX_W, MAX_H = 1200, 750   # display size; change if the window is still too big/small


def select_box(title, img):
    """Show a downscaled copy so it fits the screen, return box in FULL-size pixels."""
    scale = min(1.0, MAX_W / img.shape[1], MAX_H / img.shape[0])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    x, y, w, h = cv2.selectROI(title, small)
    cv2.destroyAllWindows()
    return int(x / scale), int(y / scale), int(w / scale), int(h / scale)


root = sys.argv[1] if len(sys.argv) > 1 else "calib"
chart = {}
for ph_dir in sorted(os.listdir(root), key=float):
    labs = []
    for name in sorted(os.listdir(os.path.join(root, ph_dir))):
        img = cv2.imread(os.path.join(root, ph_dir, name))
        if img is None:
            continue
        x, y, w, h = select_box(f"pH {ph_dir} - {name}: select WHITE area, ENTER", img)
        wb = [x, y, x + w, y + h] if w and h else None
        bal = white_balance(img, wb)
        x, y, w, h = select_box(f"pH {ph_dir} - {name}: select COLOR PAD, ENTER", bal)
        if not (w and h):
            continue
        # same center-crop + median as the server uses, so calibration matches analysis
        rgb, _ = pad_color(bal, [x, y, x + w, y + h])
        labs.append(rgb_to_lab(rgb))
    if labs:
        mean = np.mean(labs, axis=0)
        spread = np.mean(np.linalg.norm(np.array(labs) - mean, axis=1)) if len(labs) > 1 else 0
        chart[str(float(ph_dir))] = mean.round(2).tolist()
        print(f"pH {ph_dir}: {len(labs)} photos, Lab {mean.round(1)}, photo-to-photo spread {spread:.1f}")

with open(CHART_PATH, "w") as f:
    json.dump(chart, f, indent=2)
print("Saved", CHART_PATH, "- now upload it to your Hugging Face Space.")
