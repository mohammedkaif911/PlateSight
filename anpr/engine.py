"""Public API: PlateRecognizer.

Ties the pipeline together:
    OCR text -> align (detect hidden chars) -> generate candidates
             -> match mock DB -> score -> ranked predictions.

In Phase B the CV stage (YOLO + EasyOCR) simply feeds `ocr_text`,
`vehicle_type` and `vehicle_color` into `identify()`.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import database as dbmod
from .alignment import align, Alignment
from .candidate_gen import generate_candidates
from .plate_format import normalize, pretty, decode_plate
from .scoring import score_candidate


@dataclass
class Prediction:
    plate: str
    pretty: str
    confidence: float
    in_database: bool
    vehicle_type: Optional[str] = None
    color: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    owner_name: Optional[str] = None
    owner_age: Optional[int] = None
    owner_address: Optional[str] = None
    state_name: Optional[str] = None
    district_name: Optional[str] = None
    breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecognitionResult:
    best: Optional[Prediction]
    ranked: List[Prediction]
    missing_positions: List[int]
    missing_count: int
    pattern: str          # OCR aligned to template, with '?' for hidden cells
    reliable: bool
    ocr_text: str
    via_relaxation: bool = False   # True if recovered via DB-guided relaxation

    def summary(self) -> str:
        if not self.reliable:
            return (f"OCR '{self.ocr_text}' -> pattern '{self.pattern}': "
                    f"{self.missing_count} characters hidden -> UNRELIABLE "
                    f"(too many missing to predict).")
        b = self.best
        if b is None:
            return f"OCR '{self.ocr_text}' -> no candidate could be formed."
        head = (f"OCR '{self.ocr_text}' -> pattern '{self.pattern}' "
                f"({self.missing_count} hidden)")
        body = (f"  PREDICTED: {b.pretty}  [{b.plate}]  "
                f"confidence={b.confidence}%  in_db={b.in_database}")
        if b.in_database:
            body += f"  type={b.vehicle_type} color={b.color} brand={b.brand}"
        return head + "\n" + body


class PlateRecognizer:
    def __init__(self, db_path: str = "data/mock_vehicles.db", n: int = 500,
                 seed: int = 42, auto_build: bool = True, max_candidates: int = 2000,
                 max_missing: int = 2, relax: bool = True):
        self.db_path = db_path
        self.max_candidates = max_candidates
        self.max_missing = max_missing
        self.relax = relax
        if auto_build and not os.path.exists(db_path):
            dbmod.build_database(db_path, n=n, seed=seed)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._plate_cache = self._load_plate_cache()

    def _load_plate_cache(self):
        """Load all plates into a set for O(1) in-memory lookup."""
        import sqlite3 as _s
        cur = self.conn.cursor()
        cur.execute("SELECT plate FROM vehicles")
        return {row[0] for row in cur.fetchall()}

    # allow use as a context manager
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.conn.close()

    def identify(self, ocr_text: str, vehicle_type: Optional[str] = None,
                 vehicle_color: Optional[str] = None,
                 vehicle_model: Optional[str] = None,
                 top_k: int = 5) -> RecognitionResult:
        ocr = normalize(ocr_text)
        alignment = align(ocr)
        candidates, total_missing, overflow = generate_candidates(alignment, self.max_candidates)

        # Too many hidden characters -> cannot predict reliably (honest, per report).
        if total_missing > self.max_missing or overflow:
            return RecognitionResult(
                best=None, ranked=[],
                missing_positions=alignment.missing_positions,
                missing_count=total_missing,
                pattern=alignment.pattern_string(),
                reliable=False, ocr_text=ocr,
            )

        scored: List[Prediction] = []
        for cand in candidates:
            rec = dbmod.query(self.conn, cand) if cand in self._plate_cache else None
            conf, breakdown = score_candidate(
                cand, alignment.shape, alignment, rec,
                vehicle_type, vehicle_color, vehicle_model)
            dec = decode_plate(cand) or {}
            scored.append(Prediction(
                plate=cand, pretty=pretty(cand, alignment.shape),
                confidence=conf, in_database=rec is not None,
                vehicle_type=rec["vehicle_type"] if rec else None,
                color=rec["color"] if rec else None,
                brand=rec["brand"] if rec else None,
                model=rec["model"] if rec else None,
                owner_name=rec.get("owner_name") if rec else None,
                owner_age=rec.get("owner_age") if rec else None,
                owner_address=rec.get("owner_address") if rec else None,
                state_name=dec.get("state_name"),
                district_name=dec.get("district_name"),
                breakdown=breakdown,
            ))

        scored.sort(key=lambda p: p.confidence, reverse=True)
        ranked = scored[:top_k]

        via_relax = False
        # DB-guided relaxation: if the straight pass found no DB match, the OCR
        # likely misread a visible character. Try treating each confident cell
        # as hidden, one at a time, and keep any DB match found.
        if self.relax and not (ranked and ranked[0].in_database):
            relaxed = self._relax_search(alignment, vehicle_type,
                                         vehicle_color, vehicle_model)
            if relaxed:
                ranked = relaxed[:top_k]
                via_relax = True

        return RecognitionResult(
            best=ranked[0] if ranked else None,
            ranked=ranked,
            missing_positions=alignment.missing_positions,
            missing_count=total_missing,
            pattern=alignment.pattern_string(),
            reliable=True, ocr_text=ocr, via_relaxation=via_relax,
        )

    def _gen_for(self, alignment: Alignment, relax_idxs, vt, vc, vm,
                 cap) -> List[Prediction]:
        """Generate DB-hit predictions after forcing the given cells to hidden."""
        kinds2 = list(alignment.kinds)
        relaxed_m = 0
        for i in relax_idxs:
            if kinds2[i] == "M":
                relaxed_m += 1
            kinds2[i] = "X"
        aln2 = Alignment(
            shape=alignment.shape, cells=list(alignment.cells), kinds=kinds2,
            matched=max(0, alignment.matched - relaxed_m), soft=alignment.soft,
            missing=alignment.missing + len(relax_idxs), wrong=alignment.wrong,
            extra=alignment.extra)
        cands, hidden, overflow = generate_candidates(aln2, cap)
        if overflow or hidden > self.max_missing:
            return []
        out: List[Prediction] = []
        for cand in cands:
            rec = dbmod.query(self.conn, cand) if cand in self._plate_cache else None
            if rec:
                conf, brk = score_candidate(cand, aln2.shape, aln2, rec, vt, vc, vm)
                dec = decode_plate(cand) or {}
                out.append(Prediction(
                    plate=cand, pretty=pretty(cand, aln2.shape), confidence=conf,
                    in_database=True, vehicle_type=rec["vehicle_type"],
                    color=rec["color"], brand=rec["brand"], model=rec["model"],
                    owner_name=rec.get("owner_name"), owner_age=rec.get("owner_age"),
                    owner_address=rec.get("owner_address"),
                    state_name=dec.get("state_name"), district_name=dec.get("district_name"),
                    breakdown=brk))
        return out

    def _relax_search(self, alignment: Alignment, vehicle_type, vehicle_color,
                      vehicle_model) -> List[Prediction]:
        """DB-guided correction of misreads the confusion table cannot fix.
        First relax one confident cell at a time; if that fails, relax pairs of
        cells within the same segment (bounded) to handle two simultaneous
        misreads."""
        ms = [i for i, k in enumerate(alignment.kinds) if k in ("M", "S")]
        for i in ms:                                  # single-cell relaxation
            hits = self._gen_for(alignment, [i], vehicle_type, vehicle_color,
                                 vehicle_model, self.max_candidates)
            if hits:
                return sorted(hits, key=lambda p: p.confidence, reverse=True)
        from itertools import combinations
        seg_of = {}
        for si, (s, e) in enumerate(alignment.shape.segment_bounds()):
            for p in range(s, e):
                seg_of[p] = si
        cap = min(4000, self.max_candidates)          # two-cell relaxation (bounded)
        for i, j in combinations(ms, 2):
            if seg_of.get(i) != seg_of.get(j):
                continue
            hits = self._gen_for(alignment, [i, j], vehicle_type, vehicle_color,
                                 vehicle_model, cap)
            if hits:
                return sorted(hits, key=lambda p: p.confidence, reverse=True)
        return []
