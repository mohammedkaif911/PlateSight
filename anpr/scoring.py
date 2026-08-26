"""Confidence scoring for candidate plates.

Transparent, weighted blend (max 100):
    OCR alignment quality   30   (matched cells / total cells)
    Valid state code        10
    Database match          30
    Vehicle-type match      10
    Color match              5   (weakest signal)
    Vehicle model/make match 15  (strong, specific signal)

A plate with no database hit can score at most 40 (format-only guess), which
keeps "no evidence" predictions honestly low-confidence.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .alignment import Alignment
from .plate_format import PlateShape, VALID_STATES

W_ALIGN, W_STATE = 30, 10
W_DB, W_TYPE, W_COLOR, W_MODEL = 30, 10, 5, 15


def _model_matches(input_model: Optional[str], record: Optional[Dict]) -> bool:
    if not input_model or record is None:
        return False
    hay = f"{record.get('brand', '')} {record.get('model', '')}".lower().strip()
    needle = input_model.lower().strip()
    if not needle:
        return False
    return needle in hay or hay in needle


def score_candidate(candidate: str, shape: PlateShape, alignment: Alignment,
                    db_record: Optional[Dict], vehicle_type: Optional[str] = None,
                    vehicle_color: Optional[str] = None,
                    vehicle_model: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
    total = shape.total or 1
    breakdown: Dict[str, Any] = {}
    score = 0.0

    align_pts = W_ALIGN * (alignment.matched / total)
    score += align_pts
    breakdown["ocr_alignment"] = round(align_pts, 1)

    state_ok = candidate[:2] in VALID_STATES
    breakdown["state_valid"] = W_STATE if state_ok else 0
    score += breakdown["state_valid"]

    if db_record is not None:
        breakdown["database_match"] = W_DB
        score += W_DB
        type_pts = W_TYPE if (vehicle_type and db_record.get("vehicle_type") == vehicle_type) else 0
        color_pts = W_COLOR if (vehicle_color and db_record.get("color") == vehicle_color) else 0
        model_pts = W_MODEL if _model_matches(vehicle_model, db_record) else 0
        breakdown["type_match"] = type_pts
        breakdown["color_match"] = color_pts
        breakdown["model_match"] = model_pts
        score += type_pts + color_pts + model_pts
    else:
        breakdown["database_match"] = 0
        breakdown["type_match"] = 0
        breakdown["color_match"] = 0
        breakdown["model_match"] = 0

    return round(min(score, 100.0), 1), breakdown
