"""FastAPI backend — serves a custom Google-style frontend and the recovery engine."""
import base64
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import threading
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from anpr import PlateRecognizer
from anpr import database as dbmod

app = FastAPI(title="ANPR Partial Plate Recovery")
engine = PlateRecognizer(db_path="data/mock_vehicles.db", auto_build=True)
engine.max_missing = 4
engine.max_candidates = 30000
_ELOCK = threading.Lock()
_cv = None

# Load seed vehicles from data/seed.json (these persist across restarts)
_SEED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "seed.json")
if os.path.exists(_SEED_FILE):
    try:
        with open(_SEED_FILE) as f:
            seed_records = json.load(f)
        for rec in seed_records:
            dbmod.add_record(engine.conn, rec)
            engine._plate_cache.add(rec["plate"])
    except Exception:
        pass  # If seed file is malformed, continue with mock DB only


def get_cv():
    global _cv
    if _cv is None:
        import cv_pipeline as cvp
        _cv = cvp.CVPipeline()
    return _cv


def _pred(p):
    return {"plate": p.plate, "pretty": p.pretty, "confidence": p.confidence,
            "in_database": p.in_database, "type": p.vehicle_type, "color": p.color,
            "brand": p.brand, "model": p.model,
            "owner_name": p.owner_name, "owner_age": p.owner_age,
            "owner_address": p.owner_address,
            "state_name": p.state_name, "district_name": p.district_name,
            "breakdown": p.breakdown}


def result_dict(res):
    if res is None:
        return {"reliable": False, "pattern": "", "missing": 0,
                "via_relaxation": False, "best": None, "ranked": []}
    return {"reliable": res.reliable, "pattern": res.pattern,
            "missing": res.missing_count,
            "via_relaxation": getattr(res, "via_relaxation", False),
            "best": None if not res.best else _pred(res.best),
            "ranked": [_pred(p) for p in res.ranked]}


def _enc(bgr):
    import cv2
    if bgr is None:
        return None
    ok, buf = cv2.imencode(".jpg", bgr)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


@app.get("/", response_class=HTMLResponse)
def index():
    return open("static/index.html").read()


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/results")
def results():
    p = "results.json"
    return json.load(open(p)) if os.path.exists(p) else {}


@app.get("/api/db_count")
def db_count():
    import sqlite3
    c = sqlite3.connect("data/mock_vehicles.db").cursor()
    return {"count": c.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]}


class IdentifyIn(BaseModel):
    plate: str = ""
    type: str | None = None
    color: str | None = None
    model: str | None = None
    max_missing: int = 4
    topk: int = 5


@app.post("/api/identify")
def identify(inp: IdentifyIn):
    try:
        engine.max_missing = inp.max_missing
        with _ELOCK:
            r = engine.identify(inp.plate, vehicle_type=inp.type, vehicle_color=inp.color,
                                vehicle_model=inp.model, top_k=inp.topk)
        return result_dict(r)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


class AddIn(BaseModel):
    plate: str = ""
    type: str = ""
    color: str = ""
    brand: str = ""
    model: str = ""


@app.post("/api/add_vehicle")
def add_vehicle(inp: AddIn):
    try:
        plate = "".join(c for c in inp.plate.upper() if c.isalnum())
        if len(plate) < 5:
            return {"ok": False, "error": "invalid plate"}
        with _ELOCK:
            dbmod.add_record(engine.conn, {"plate": plate, "vehicle_type": inp.type,
                                           "color": inp.color, "brand": inp.brand,
                                           "model": inp.model})
            engine._plate_cache.add(plate)
        return {"ok": True, "plate": plate}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), vtype: str = Form(""),
                 vcolor: str = Form(""), vmodel: str = Form("")):
    try:
        # --- upload security: validate type + size before processing ---
        if not file.content_type or not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400,
                                content={"error": "Only image files are accepted."})
        data = await file.read()
        if len(data) > 10 * 1024 * 1024:  # 10 MB limit
            return JSONResponse(status_code=413,
                                content={"error": "File too large (max 10 MB)."})
        import cv2, numpy as np
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Could not read image."})
        cvres = get_cv().process(img)
        variants = cvres.ocr_variants or ([cvres.ocr_text] if cvres.ocr_text else [])
        engine.max_missing = 4
        eff_type = vtype or cvres.vehicle_type
        eff_color = vcolor or cvres.color
        best, bk, used = None, (-1, -1.0), cvres.ocr_text
        with _ELOCK:
            for t in dict.fromkeys(variants):
                r = engine.identify(t, vehicle_type=eff_type,
                                    vehicle_color=eff_color,
                                    vehicle_model=vmodel or None, top_k=5)
                rdb = 1 if (r.reliable and r.best and r.best.in_database) else 0
                conf = r.best.confidence if (r.reliable and r.best) else 0.0
                if (rdb, conf) > bk:
                    best, bk, used = r, (rdb, conf), t
        gc.collect()
        return {"ocr": cvres.ocr_text, "type": cvres.vehicle_type, "color": cvres.color,
                "notes": cvres.notes, "annotated": _enc(cvres.annotated_bgr),
                "plate_crop": _enc(cvres.plate_crop_bgr),
                "result": result_dict(best), "used": used}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})
