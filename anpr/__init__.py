"""ANPR with Partial Plate Recovery.

Phase A: the inference engine (pure Python, no heavy CV dependencies).
Given an OCR string (and optional vehicle attributes), it detects which
characters are hidden, generates plausible candidates, matches them against
a vehicle database, and returns ranked predictions with confidence scores.
"""
from .engine import PlateRecognizer, RecognitionResult, Prediction
from .plate_format import PlateShape, VALID_STATES, normalize, pretty

__all__ = [
    "PlateRecognizer",
    "RecognitionResult",
    "Prediction",
    "PlateShape",
    "VALID_STATES",
    "normalize",
    "pretty",
]
