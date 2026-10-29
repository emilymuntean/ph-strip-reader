"""
Build chart.json from real buffers - automatically, using the SAME code path the server uses
(markers -> flatten -> white balance -> fixed pad window), so calibration and patient photos match.

Folder layout (folder name = pH measured with the calibrated meter, ASTM E70 style):
    calib/6.0/a.jpg  calib/6.0/b.jpg  calib/6.2/a.jpg ...   (>= 3 photos per buffer, ideally taken
    on different days / lighting / phones - this is what makes the chart robust)

Each photo: mat with markers + white patch visible, the pad windows wet with that ONE buffer,
photographed at the SAME wait time you will use in the app (e.g. 60-180 s - pick one and keep it).

    python calibrate.py calib                       # use every pad in every photo
    python calibrate.py calib --pads "L heel,R heel"  # only pads that were actually dipped

Outputs: chart.json (commit it!) and calibration_report.csv, plus suggested DIST_* thresholds and
the 3x-spread resolution check from section 7.1/7.2 of the design history file.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

import reader


def collect(root, layout, pad_names=None):
    samples = {}
    for d in sorted(os.listdir(root)):
        try:
            ph = float(d)
        except ValueError:
            continue
        folder = os.path.join(root, d)
        for fn in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, fn))
            if img is None:
                continue
            m = reader.measure(img, layout)
            if not m["markers_found"]:
                print(f"  skip {d}/{fn}: markers not found")
                continue
            for w in m["warnings"]:
                print(f"  note {d}/{fn}: {w}")
            for p in m["pads"]:
                if "lab" not in p or (pad_names and p["name"] not in pad_names):
                    continue
                samples.setdefault(ph, []).append((fn, p["name"], p["lab"]))
    return samples


def build_chart(samples, max_dev=10.0):
    chart, rows = {}, []
    for ph in sorted(samples):
        labs = np.array([s[2] for s in samples[ph]])
        med = np.median(labs, 0)
        dev = np.linalg.norm(labs - med, axis=1)
        keep = dev <= max_dev
        mean = labs[keep].mean(0)
        spread = float(np.mean(np.linalg.norm(labs[keep] - mean, axis=1)))
        chart[ph] = mean
        rows.append({"ph": ph, "n_used": int(keep.sum()), "n_dropped": int((~keep).sum()),
                     "L": mean[0], "a": mean[1], "b": mean[2], "spread_dE": spread})
    return chart, rows


def resolution(chart, rows):
    keys = sorted(chart)
    spr = {r["ph"]: max(r["spread_dE"], 0.5) for r in rows}
    for k in range(1, 6):
        ok = all(np.linalg.norm(chart[keys[i + k]] - chart[keys[i]]) >= 3 * max(spr[keys[i]], spr[keys[i + k]])
                 for i in range(len(keys) - k))
        if ok and len(keys) > k:
            return round((keys[k] - keys[0]) / k, 2)
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="calib")
    ap.add_argument("--pads", default="")
    ap.add_argument("--layout", default=reader.LAYOUT_PATH)
    ap.add_argument("--out", default=reader.CHART_PATH)
    a = ap.parse_args()

    layout = reader.load_layout(a.layout)
    pads = [p.strip() for p in a.pads.split(",") if p.strip()] or None
    samples = collect(a.root, layout, pads)
    if len(samples) < 3:
        raise SystemExit("Need at least 3 buffer folders with readable photos.")
    chart, rows = build_chart(samples)

    print("\npH    n   spread(dE)   Lab")
    for r in rows:
        print(f"{r['ph']:<5g} {r['n_used']:<3d} {r['spread_dE']:<11.2f}  "
              f"({r['L']:.1f}, {r['a']:.1f}, {r['b']:.1f})" + (f"  [{r['n_dropped']} outliers dropped]" if r['n_dropped'] else ""))

    keys = sorted(chart)
    print("\nNeighbor separation (needs >= 3x photo-to-photo spread to tell steps apart):")
    for k0, k1 in zip(keys[:-1], keys[1:]):
        sep = float(np.linalg.norm(chart[k1] - chart[k0]))
        s = max(r["spread_dE"] for r in rows if r["ph"] in (k0, k1))
        print(f"  {k0:g}->{k1:g}: separation {sep:.1f}, spread {s:.1f}, ratio {sep / max(s, .5):.1f} "
              + ("OK" if sep >= 3 * max(s, .5) else "TOO CLOSE"))
    print(f"Estimated resolution (smallest pH spacing meeting 3x criterion): {resolution(chart, rows)}")

    # in-sample check + threshold suggestion
    dists, errs = [], []
    for ph, lst in samples.items():
        for _, _, lab in lst:
            p, d, _ = reader.lab_to_ph(lab, chart)
            dists.append(d); errs.append(p - ph)
    p95 = float(np.percentile(dists, 95))
    print(f"\nIn-sample (optimistic) pH error: MAE {np.mean(np.abs(errs)):.3f}, "
          f"95th pct distance to curve {p95:.1f}")
    print(f"Suggested: DIST_HIGH={np.ceil(p95):g}  DIST_MED={np.ceil(2 * p95):g} "
          f"(set as env vars on Render, or edit reader.py)")

    with open(a.out, "w") as f:
        json.dump({str(k): v.round(3).tolist() for k, v in chart.items()}, f, indent=2)
    with open("calibration_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {a.out} and calibration_report.csv - commit chart.json and redeploy.")
