"""
Build chart.json for YOUR strips  (run on your laptop, then upload chart.json to the Hugging Face Space)

For +/-0.2 pH accuracy, calibrate with real buffers, not the printed card:
  1. Make/buy buffers across your range, ideally every 0.5 pH (3.0, 3.5 ... 10.0). Verify each with a pH meter.
  2. Dip a pad in each buffer, photograph it ON THE MAT under the same lighting/phone you will use for patients.
     3+ photos per pH is better.
  3. Folders named by the meter reading:
       calib/6.5/a.jpg   calib/7.0/a.jpg   calib/7.5/a.jpg ...
  4. Run:  python calibrate.py calib
     For each photo two windows appear:
       (1) drag a box over a plain WHITE area of the mat -> ENTER
       (2) drag a box over the reacted color pad          -> ENTER
  5. chart.json is saved. Upload it to your Hugging Face Space (Files > Add file > Upload).

Needs: pip install opencv-python numpy fastapi python-multipart
"""
import json, os, sys
import cv2
import numpy as np
from server import white_balance, rgb_to_lab, pad_color, CHART_PATH

root = sys.argv[1] if len(sys.argv) > 1 else "calib"
chart = {}
for ph_dir in sorted(os.listdir(root), key=float):
    labs = []
    for name in sorted(os.listdir(os.path.join(root, ph_dir))):
        img = cv2.imread(os.path.join(root, ph_dir, name))
        if img is None:
            continue
        x, y, w, h = cv2.selectROI(f"pH {ph_dir} - {name}: select WHITE area, ENTER", img)
        cv2.destroyAllWindows()
        wb = [x, y, x + w, y + h] if w and h else None
        bal = white_balance(img, wb)
        x, y, w, h = cv2.selectROI(f"pH {ph_dir} - {name}: select COLOR PAD, ENTER", bal)
        cv2.destroyAllWindows()
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
