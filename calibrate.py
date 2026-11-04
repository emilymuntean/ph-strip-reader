"""
Run this on your laptop to build chart.json.
1. Create a folder called 'calib'
2. Inside 'calib', make folders for your solutions: 6.1, 6.3, 6.5 ... 7.9
3. Put the photos of the dipped strips inside those folders.
4. Run: python calibrate.py calib
5. Upload the generated 'chart.json' to GitHub/Render.
"""
import json, os, sys
import cv2
import numpy as np

def white_balance(img, white_box):
    f = img.astype(np.float32)
    x1, y1, x2, y2 = white_box
    ref = np.median(f[y1:y2, x1:x2].reshape(-1, 3), axis=0)
    f *= 240.0 / np.maximum(ref, 1)
    return np.clip(f, 0, 255).astype(np.uint8)

def rgb_to_lab(rgb):
    px = np.uint8([[np.clip(np.round(rgb), 0, 255)]])
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    return np.array([lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128])

def pad_color(img, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad = img[y1 + h // 4: y2 - h // 4, x1 + w // 4: x2 - w // 4]
    if pad.size == 0: pad = img[y1:y2, x1:x2]
    px = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(float)
    keep = ~((px.min(1) > 235))
    if keep.sum() > 20: px = px[keep]
    return np.median(px, axis=0)

root = sys.argv[1] if len(sys.argv) > 1 else "calib"
chart = {}

for ph_dir in sorted(os.listdir(root), key=float):
    labs = []
    for name in sorted(os.listdir(os.path.join(root, ph_dir))):
        img = cv2.imread(os.path.join(root, ph_dir, name))
        if img is None: continue
        
        # 1. Select White Area
        x, y, w, h = cv2.selectROI(f"pH {ph_dir} - {name}: select WHITE area", img)
        cv2.destroyAllWindows()
        if not (w and h): continue
        bal = white_balance(img, [x, y, x+w, y+h])
        
        # 2. Select Color Pad
        x, y, w, h = cv2.selectROI(f"pH {ph_dir} - {name}: select COLOR PAD", bal)
        cv2.destroyAllWindows()
        if not (w and h): continue
        
        rgb = pad_color(bal, [x, y, x+w, y+h])
        labs.append(rgb_to_lab(rgb))
        
    if labs:
        mean = np.mean(labs, axis=0)
        chart[str(float(ph_dir))] = mean.round(2).tolist()
        print(f"pH {ph_dir}: {len(labs)} photos, Lab {mean.round(1)}")

with open("chart.json", "w") as f:
    json.dump(chart, f, indent=2)
print("Saved chart.json! Upload this to GitHub.")
