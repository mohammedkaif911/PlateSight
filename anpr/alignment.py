"""Align an OCR string to the Indian plate template (the project's "brain").

For every plate shape we run a DP alignment between the OCR string and the
template cells. Each cell is either filled by an OCR character or left HIDDEN.
Crucially, a character of the WRONG class that is a known visual confusion
(e.g. 'Z' read where a '2' was, 'I' for '1', 'O' for '0') is treated as a
SOFT (likely-misread) cell with a tiny candidate set -- not a hard mismatch.
This is what makes the system robust to real OCR errors, not just hidden chars.

Cell kinds:  M = matched (confident)   S = soft (confusable misread)
             X = hidden (missing)      W = wrong class, not confusable
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .plate_format import PlateShape, all_shapes, CLASS_LETTER

MATCH_COST = 0       # OCR char fills a cell of the correct class
SUBST_COST = 1       # confusable misread, e.g. 'Z' where a '2' was
EXTRA_COST = 1       # OCR noise / separator, dropped
MISSING_COST = 1     # a plate cell is hidden / unreadable
MISMATCH_COST = 5    # non-confusable wrong class (rare)

# Visual confusion pairs (letter, digit).
_PAIRS = [
    ("O", "0"), ("I", "1"), ("L", "1"), ("S", "5"), ("B", "8"),
    ("Z", "2"), ("G", "6"), ("G", "9"), ("D", "0"), ("Q", "0"), ("T", "7"),
]


def _class_of(c: str) -> str:
    return CLASS_LETTER if c.isalpha() else "D"


def confusable_targets(c: str, want: str) -> set:
    """Characters of class `want` that `c` is commonly confused with."""
    out = set()
    for ltr, dig in _PAIRS:
        members = {ltr, dig}
        if c in members:
            for m in members:
                if m != c and _class_of(m) == want:
                    out.add(m)
    return out


def _match_cost_and_kind(c: str, klass: str):
    if _class_of(c) == klass:
        return MATCH_COST, "M"
    if confusable_targets(c, klass):
        return SUBST_COST, "S"
    return MISMATCH_COST, "W"


def _shape_prior(shape: PlateShape) -> int:
    return 2 if shape.series_n == 1 else 0


@dataclass
class Alignment:
    shape: PlateShape
    cells: List[Optional[str]]
    kinds: List[str] = field(default_factory=list)
    cost: float = 0.0
    matched: int = 0
    soft: int = 0
    missing: int = 0
    wrong: int = 0
    extra: int = 0

    @property
    def missing_positions(self) -> List[int]:
        return [i for i, k in enumerate(self.kinds) if k in ("X", "W")]

    @property
    def missing_count(self) -> int:
        return sum(1 for k in self.kinds if k == "X")

    def pattern_string(self) -> str:
        out = []
        for c, k in zip(self.cells, self.kinds):
            if k in ("M", "S"):
                out.append(c.lower() if k == "S" else c)  # lowercase = soft/misread
            else:
                out.append("?")
        return "".join(out)


def _align_shape(ocr: str, shape: PlateShape) -> Alignment:
    m, n = len(ocr), shape.total
    klasses = shape.klasses()
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]

    dp[0][0] = 0
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + MISSING_COST
        back[0][j] = "missing"
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + EXTRA_COST
        back[i][0] = "extra"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            char = ocr[i - 1]
            match_cost, _ = _match_cost_and_kind(char, klasses[j - 1])

            best_move, best_cost = "match", dp[i - 1][j - 1] + match_cost
            missing_cost = dp[i][j - 1] + MISSING_COST
            if missing_cost < best_cost:
                best_move, best_cost = "missing", missing_cost
            extra_cost = dp[i - 1][j] + EXTRA_COST
            if extra_cost < best_cost:
                best_move, best_cost = "extra", extra_cost

            dp[i][j] = best_cost
            back[i][j] = best_move

    cells: List[Optional[str]] = [None] * n
    kinds: List[str] = ["X"] * n
    matched = soft = missing = wrong = extra = 0
    i, j = m, n
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "match":
            char = ocr[i - 1]
            _, kind = _match_cost_and_kind(char, klasses[j - 1])
            cells[j - 1] = char
            kinds[j - 1] = kind
            if kind == "M":
                matched += 1
            elif kind == "S":
                soft += 1
            else:
                wrong += 1
            i -= 1
            j -= 1
        elif move == "missing":
            missing += 1
            j -= 1
        else:
            extra += 1
            i -= 1

    return Alignment(shape=shape, cells=cells, kinds=kinds, cost=dp[m][n],
                     matched=matched, soft=soft, missing=missing,
                     wrong=wrong, extra=extra)


def align(ocr_text: str, shapes=None) -> Alignment:
    ocr = ocr_text
    best: Optional[Alignment] = None
    for shape in (shapes or all_shapes()):
        cand = _align_shape(ocr, shape)
        cand.cost += _shape_prior(shape)
        if best is None or cand.cost < best.cost:
            best = cand
    return best  # type: ignore[return-value]
