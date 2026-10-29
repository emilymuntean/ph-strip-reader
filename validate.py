"""
Accuracy test on photos that were NOT used for calibration (design spec: within +/-0.2 pH).

    test/6.4/a.jpg  test/7.1/b.jpg ...   (folder name = calibrated-meter pH)
    python validate.py test [--pads "L heel,R heel"]

Reports MAE, bias, SD, % within +/-0.2, and how many pads the app flagged for review.
Writes validation_results.csv for your lab notebook / section 7.4.
"""
import argparse
import csv
import os

import cv2
import numpy as np

import reader

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default="test")
ap.add_argument("--pads", default="")
a = ap.parse_args()
pads = [p.strip() for p in a.pads.split(",") if p.strip()] or None
layout, chart = reader.load_layout(), reader.load_chart()
if chart is None:
    raise SystemExit("chart.json not found - run calibrate.py first")

rows = []
for d in sorted(os.listdir(a.root)):
    try:
        truth = float(d)
    except ValueError:
        continue
    for fn in sorted(os.listdir(os.path.join(a.root, d))):
        img = cv2.imread(os.path.join(a.root, d, fn))
        if img is None:
            continue
        res = reader.analyze(img, fn, layout, chart)
        for s in res["strips"]:
            if pads and s["location"] not in pads:
                continue
            if s["raw_ph"] is None:
                rows.append([truth, fn, s["location"], "", "", s["status"], s["note"]])
                continue
            rows.append([truth, fn, s["location"], s["raw_ph"], round(s["raw_ph"] - truth, 2),
                         s["status"], s["note"]])

with open("validation_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["true_pH", "photo", "pad", "app_pH", "error", "status", "note"])
    w.writerows(rows)

read = [r for r in rows if r[4] != ""]
ok = [r for r in read if r[5] == "ok"]


def stats(rs, label):
    if not rs:
        print(f"{label}: no readings"); return
    e = np.array([r[4] for r in rs], float)
    print(f"{label}: n={len(e)}  MAE {np.mean(np.abs(e)):.3f}  bias {e.mean():+.3f}  SD {e.std():.3f}  "
          f"within +/-0.2: {100 * np.mean(np.abs(e) <= 0.2):.1f}%  max |err| {np.abs(e).max():.2f}")


stats(read, "all reads      ")
stats(ok, "status=ok only  ")
print(f"flagged needs_review: {len(rows) - len(ok)} of {len(rows)} pad readings")
print("Saved validation_results.csv")
