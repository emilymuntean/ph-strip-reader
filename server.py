import os
import json
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
import reader

app = FastAPI(title="pH Strip Reader API (n8n Gemini)")

@app.get("/health")
def health():
    return {"ok": True, "version": "5.0 (n8n-gemini)"}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    id: str = Form(None),
    boxes: str = Form(...), # This will now hold the JSON from n8n
    debug: str = Form("false"),
):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None: return {"error": "Could not read image"}

    try:
        # Parse the boxes JSON sent from n8n
        loc = json.loads(boxes)
    except Exception as e:
        return {"error": f"Failed to parse boxes JSON: {e}"}

    # Pass the pre-calculated boxes directly to a new reader function
    try:
        res = reader.analyze_with_boxes(img, loc, rec_id=id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Server Error: {type(e).__name__}: {str(e)}"}

    wb = res.pop("_wb_image", None)
    if loc is not None:
        try: res["annotated_image_base64"] = reader.encode_b64(reader.annotate(img, loc))
        except Exception: pass
    if str(debug).lower() == "true" and wb is not None:
        res["flat_image_base64"] = reader.encode_b64((wb[:, :, ::-1] * 255).astype(np.uint8))
    
    return res
