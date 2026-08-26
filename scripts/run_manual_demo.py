"""Prove the engine recovers the user's REAL plates via manual entry
(bypassing the low-res OCR problem)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anpr import PlateRecognizer
from anpr import database as dbmod

e = PlateRecognizer(db_path="data/mock_vehicles.db", auto_build=True)
e.max_missing = 4
e.max_candidates = 100000
dbmod.add_record(e.conn, {"plate": "KL02BP7403", "vehicle_type": "motorcycle",
                          "color": "blue", "brand": "", "model": ""})
dbmod.add_record(e.conn, {"plate": "TN02AK8055", "vehicle_type": "car",
                          "color": "black", "brand": "BMW", "model": "320b"})

# (manual partial plate, type, color, model, full truth, what's hidden)
TESTS = [
    ("KL02BP74",  "motorcycle", "blue",  None,   "KL02BP7403", "last 2 digits (03)"),
    ("KL02BP740", "motorcycle", "blue",  None,   "KL02BP7403", "last digit (3)"),
    ("TN02AK80",  "car",        "black", "320b", "TN02AK8055", "last 2 digits (55)"),
    ("TN02AK55",  "car",        "black", "320b", "TN02AK8055", "district+number trickier"),
]

print("=" * 78)
print(" MANUAL-ENTRY RECOVERY on your real plates (engine only, no OCR)")
print("=" * 78)
for ocr, vt, vc, vm, gt, note in TESTS:
    res = e.identify(ocr, vehicle_type=vt, vehicle_color=vc, vehicle_model=vm, top_k=3)
    b = res.best
    ok = "OK " if (res.reliable and b and b.plate == gt) else "XX "
    line = f"  {ok}'{ocr}' ({note:28s}) -> "
    if res.reliable and b:
        line += f"{b.pretty} [{b.plate}] conf={b.confidence}% db={b.in_database} model={b.model}"
    else:
        line += f"UNRELIABLE (pattern {res.pattern})"
    print(line)
