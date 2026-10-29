"""
pH Strip Reader API v3 - thin FastAPI wrapper around reader.py

POST /analyze (multipart form)
    file   : photo of the whole mat (all 4 corner markers visible)   [required]
    id     : record id, passed through
    debug  : "true" to also return the flattened/white-balanced mat image
    (expected / boxes are accepted and ignored so the old n8n node keeps working)

GET /health : shows whether chart.json and layout.json are loaded.

The pH number is ALWAYS computed here with color math. No AI model reads or guesses pH.
"""
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

import reader

app = FastAPI(title="pH Strip Reader v3")


@app.get("/health")
def health():
    chart = reader.load_chart()
    layout = reader.load_json(reader.LAYOUT_PATH)
    return {
        "ok": True,
        "version": "3",
        "chart_loaded": chart is not None,
        "chart_points": None if chart is None else sorted(chart),
        "layout_loaded": layout is not None,
        "pads": None if layout is None else [p["name"] for p in layout["pads"]],
        "thresholds": {"DIST_HIGH": reader.DIST_HIGH, "DIST_MED": reader.DIST_MED,
                       "SPREAD_MAX": reader.SPREAD_MAX},
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), id: str = Form(None),
                  expected: str = Form(None), boxes: str = Form(None),
                  debug: str = Form("false")):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not read image"}
    try:
        res = reader.analyze(img, rec_id=id)
    except Exception as e:  # return a readable error to n8n instead of a 500
        return {"error": f"{type(e).__name__}: {e}"}
    res["annotated_image_base64"] = reader.encode_b64(res.pop("_annotated"))
    flat = res.pop("_flat")
    if str(debug).lower() == "true" and flat is not None:
        res["flat_image_base64"] = reader.encode_b64(flat)
    return res
