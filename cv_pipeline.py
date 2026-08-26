"""Computer-vision front-end for the ANPR engine.

vehicle-YOLOv8 (COCO)  -> type (only when confident) + dominant colour
plate-YOLOv8 (your trained model, with a pretrained fallback) -> plate box
fast-plate-ocr          -> plate text + per-char confidence
If no plate is localised (common on bikes), OCR the vehicle region as a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import gc

import numpy as np

PLATE_MODEL_REPO = "Koushim/yolov8-license-plate-detection"
PLATE_MODEL_FILENAME = "best.pt"
YOLO_TYPE_MAP = {"car": "car", "motorcycle": "motorcycle", "bus": "bus", "truck": "truck"}

_HUE_COLORS = [(10, "red"), (25, "orange"), (40, "yellow"), (75, "green"),
               (100, "cyan"), (135, "blue"), (165, "magenta"), (180, "red")]


def cv_available() -> bool:
    try:
        import fast_plate_ocr  # noqa: F401
        import onnxruntime  # noqa: F401
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class CVResult:
    vehicle_type: Optional[str] = None
    color: Optional[str] = None
    ocr_text: str = ""
    skeleton: str = ""
    ocr_variants: List[str] = field(default_factory=list)
    annotated_bgr: Optional[np.ndarray] = None
    plate_crop_bgr: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)


def dominant_color(bgr_crop: np.ndarray) -> Optional[str]:
    import cv2
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    img = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = img.reshape(-1, 3).mean(axis=0)
    if mean_v < 50:
        return "black"
    if mean_s < 35 and mean_v > 180:
        return "white"
    if mean_s < 35:
        return "silver"
    for bound, name in _HUE_COLORS:
        if mean_h <= bound:
            return name
    return "unknown"


class CVPipeline:
    def __init__(self, device: str = "cpu"):
        import fast_plate_ocr as fpo
        from ultralytics import YOLO
        self.device = device
        self.vehicle_model = YOLO("yolov8n.pt")
        # General pretrained detector as the single plate model: it reliably finds
        # BOTH car and bike plates. (Your trained models/plate.pt reached 99.5% mAP
        # but was trained on a car-heavy set, so it misses two-wheelers; kept as an
        # achievement. A single model also keeps memory low -> no crashes.)
        self.plate_model = (self._load_at("plate_koushim_backup.pt")
                            or self._load_plate_model())
        self.plate_fallback = None
        self.ocr = fpo.LicensePlateRecognizer(
            hub_ocr_model="cct-s-v2-global-model", device=device)

    def _load_plate_model(self):
        import os
        import shutil
        from ultralytics import YOLO
        local = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "plate.pt"))
        if os.path.exists(local):
            return YOLO(local)
        try:
            from huggingface_hub import hf_hub_download
            os.makedirs(os.path.dirname(local), exist_ok=True)
            d = hf_hub_download(repo_id=PLATE_MODEL_REPO, filename=PLATE_MODEL_FILENAME)
            shutil.copy(d, local)
            return YOLO(local)
        except Exception:
            return None

    def _load_at(self, name):
        import os
        from ultralytics import YOLO
        p = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", name))
        return YOLO(p) if os.path.exists(p) else None

    def _largest(self, model, img, whitelist=None):
        """Largest-area box (used for plates)."""
        res = model(img, verbose=False)[0]
        names = res.names
        best, best_area = None, 0
        for box in res.boxes:
            name = names[int(box.cls[0])]
            if whitelist and name not in whitelist:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best, best_area = (x1, y1, x2, y2), area
        return best

    def _best_conf(self, model, img, whitelist=None, thr=0.0):
        """Highest-confidence box above thr -> (box, name, conf)."""
        res = model(img, verbose=False)[0]
        names = res.names
        best, bc, bn = None, thr, None
        for box in res.boxes:
            name = names[int(box.cls[0])]
            if whitelist and name not in whitelist:
                continue
            c = float(box.conf[0])
            if c > bc:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best, bc, bn = (x1, y1, x2, y2), c, name
        return best, bn, bc

    def _plate_crop(self, img, box, ar=3.0, padx=0.12, pady=0.30):
        H, W = img.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cur = bw / max(1, bh)
        if cur > ar:
            nw = int(bh * ar); cx = (x1 + x2) // 2; x1, x2 = cx - nw // 2, cx + nw // 2
        elif cur < ar:
            nh = int(bw / ar); cy = (y1 + y2) // 2; y1, y2 = cy - nh // 2, cy + nh // 2
        pw, ph = (x2 - x1) * padx, (y2 - y1) * pady
        return img[int(max(0, y1 - ph)):int(min(H, y2 + ph)),
                   int(max(0, x1 - pw)):int(min(W, x2 + pw))]

    def _ocr_crop(self, crop):
        if crop is None or crop.size == 0:
            return "", ""
        pred = self.ocr.run(crop, return_confidence=True)[0]
        text = pred.plate
        L = len(text)
        probs = list(pred.char_probs)[:L]
        skel = "".join(text[i] for i in range(L)
                       if i < len(probs) and probs[i] >= 0.5) if probs else text
        return text, skel

    def _ocr_plate(self, crop):
        """OCR a plate crop, returning several reading variants — including a
        two-line split (top + bottom concatenated) for standard Indian plates
        where state+district sit above series+number. The app tries all
        variants and keeps the best database match."""
        variants = []
        text_w, skel_w = self._ocr_crop(crop)
        if text_w:
            variants += [text_w, skel_w]
        h, w = crop.shape[:2]
        if h > 0 and w / h < 3.3 and len(text_w) < 7:  # tall crop AND short read -> missed a line
            mid = h // 2
            tt, _ = self._ocr_crop(crop[:mid])
            tb, _ = self._ocr_crop(crop[mid:])
            if tt and tb:
                variants += [tt + tb, tb + tt]
        seen, out = set(), []
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def process(self, img: np.ndarray) -> CVResult:
        import cv2
        res = CVResult(annotated_bgr=img.copy())

        # --- vehicle: highest-confidence; type only when confident (no wrong guesses) ---
        vbox, vname, vconf = self._best_conf(self.vehicle_model, img,
                                             set(YOLO_TYPE_MAP), thr=0.25)
        vcrop = None
        if vbox is not None:
            x1, y1, x2, y2 = vbox
            if vconf >= 0.45:
                res.vehicle_type = YOLO_TYPE_MAP.get(vname)
            res.color = dominant_color(img[y1:y2, x1:x2])
            vcrop = img[y1:y2, x1:x2]
            cv2.rectangle(res.annotated_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(res.annotated_bgr, f"{res.vehicle_type or 'vehicle'}/{res.color}",
                        (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- plate: general pretrained detector (finds car and bike plates) ---
        pbox = self._largest(self.plate_model, img) if self.plate_model is not None else None

        if pbox is not None:
            crop = self._plate_crop(img, pbox)
            res.plate_crop_bgr = crop
            x1, y1, x2, y2 = pbox
            cv2.rectangle(res.annotated_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
            res.ocr_variants = self._ocr_plate(crop)
        elif vcrop is not None:
            # Bike / missed-plate fallback: read text straight from the vehicle region.
            t, _ = self._ocr_crop(vcrop)
            res.ocr_variants = [t] if t else []
            res.notes.append("Plate not localised — read from vehicle region (bike fallback).")
        else:
            res.notes.append("No number plate detected.")

        if res.ocr_variants:
            res.ocr_text = res.ocr_variants[0]
            res.skeleton = res.ocr_variants[0]
        return res
