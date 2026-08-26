"""Generate plausible full plate strings for hidden AND misread positions.

Per segment we combine:
  * matched cells (fixed),
  * soft cells (a tiny set of confusable alternatives),
  * hidden/wrong cells (fillers from the full alphabet).

Hidden fillers are placed at every possible position (subsequence style) so the
system recovers a hidden digit whether it was at the front or back of a number.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import List, Set, Tuple

from .alignment import Alignment, confusable_targets
from .plate_format import CLASS_LETTER, VALID_STATES

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _is_subsequence(small: str, big: str) -> bool:
    it = iter(big)
    return all(ch in it for ch in small)


def state_candidates(cells: List[str], kinds: List[str]) -> Set[str]:
    """Valid state codes whose letters contain the matched letters in order
    (subsequence). This recovers a hidden state letter whether it was the first
    or second character (e.g. seeing only 'K' matches KA, KL, JK...)."""
    matched = [cells[i] for i in range(len(cells)) if kinds[i] == "M"]
    return {st for st in VALID_STATES if _is_subsequence(matched, st)}


def segment_candidates(cells: List[str], kinds: List[str], alphabet: str,
                       seg_len: int, expected_klass: str) -> Set[str]:
    skeleton, k = [], 0
    for c, kd in zip(cells, kinds):
        if kd == "M":
            skeleton.append(("M", c))
        elif kd == "S":
            tgt = confusable_targets(c, expected_klass) or set(alphabet)
            skeleton.append(("S", tgt))
        else:  # X or W -> filler
            k += 1
    n = len(skeleton)
    results: Set[str] = set()
    for fill_positions in combinations(range(seg_len), k):
        fill_set = set(fill_positions)
        slots = [None] * seg_len
        si, ok = 0, True
        for p in range(seg_len):
            if p not in fill_set:
                if si < n:
                    slots[p] = skeleton[si]
                    si += 1
                else:
                    ok = False
                    break
        if not ok or si != n:
            continue
        opts = []
        for p in range(seg_len):
            if p in fill_set:
                opts.append(list(alphabet))
            else:
                kind, val = slots[p]
                opts.append([val] if kind == "M" else sorted(val))
        for combo in product(*opts):
            results.add("".join(combo))
    return results


def generate_candidates(alignment: Alignment,
                        max_candidates: int = 2000
                        ) -> Tuple[List[str], int, bool]:
    shape = alignment.shape
    cells, kinds, klasses = alignment.cells, alignment.kinds, shape.klasses()
    seg = [(cells[s:e], kinds[s:e], klasses[s])
           for s, e in shape.segment_bounds()]

    def needs_gen(sk):
        return any(k in ("X", "W", "S") for k in sk)

    opts_list: List[Set[str]] = []
    # state segment (special: constrained to valid codes)
    sc, sk, _ = seg[0]
    if needs_gen(sk):
        opts_list.append(state_candidates(sc, sk))
    else:
        opts_list.append({"".join(c for c in sc if c)})

    # district / series / number
    for sc, sk, ek in seg[1:]:
        if needs_gen(sk):
            alphabet = DIGITS if ek == "D" else LETTERS
            opts_list.append(segment_candidates(sc, sk, alphabet, len(sc), ek))
        else:
            opts_list.append({"".join(c for c in sc if c)})

    total_hidden = sum(1 for k in kinds if k == "X")

    results: Set[str] = set()
    for combo in product(*opts_list):
        results.add("".join(combo))
        if len(results) > max_candidates:
            return sorted(results), total_hidden, True
    return sorted(results), total_hidden, False
