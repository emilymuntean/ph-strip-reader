"""
pH Strip Reader API v4 - card-based calibration.

POST /analyze (multipart form)
    file   : photo of the whole mat (strips + reference card + white area visible) [required]
    id     : record id, passed through
    debug  : "true" to also return the white-balanced mat image

GET /health : shows server version and whether GEMINI_API_KEY is set.

The pH number is computed here in Python using the reference card in the
SAME photo. Gemini only locates boxes and reads the printed pH labels on the card.
"""
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

import reader

app = FastAPI(title="pH Strip Reader v4")


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": "4",
        "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY")),
        "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        "thresholds": {
            "DIST_HIGH": reader.DIST_HIGH,
            "DIST_MED": reader.DIST_MED,
        },
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    id: str = Form(None),
    debug: str = Form("false"),
):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not read image"}

    try:
        res = reader.analyze(img, rec_id=id)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    wb = res.pop("_wb_image", None)
    loc = res.pop("_locate", None)
    if loc is not None:
        try:
            res["annotated_image_base64"] = reader.encode_b64(
                reader.annotate(img, loc)
            )
        except Exception:
            pass
    if str(debug).lower() == "true" and wb is not None:
        res["flat_image_base64"] = reader.encode_b64(
            (wb[:, :, ::-1] * 255).astype(np.uint8)
        )
    return res