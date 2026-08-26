"""Quick visual demo of the inference engine on realistic scenarios."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anpr import PlateRecognizer
from anpr import database as dbmod

# A realistic mock fleet the system "knows about".
FLEET = [
    {"plate": "KA51AB1234", "vehicle_type": "car",       "color": "black",  "brand": "Maruti Suzuki"},
    {"plate": "KA05MN9876", "vehicle_type": "motorcycle","color": "red",    "brand": "Bajaj"},
    {"plate": "KA53XY0001", "vehicle_type": "car",       "color": "white",  "brand": "Hyundai"},
    {"plate": "MH12DE4567", "vehicle_type": "suv",       "color": "blue",   "brand": "Mahindra"},
    {"plate": "TN07FG8910", "vehicle_type": "car",       "color": "silver", "brand": "Honda"},
]

SCENARIOS = [
    ("KA51AB1234",          "car",        "black",   "Perfect plate read"),
    ("ka 51 ab 1234",       "car",        "black",   "Same plate, messy OCR (spaces/case)"),
    ("KA51AB123",           "car",        "black",   "Hidden LAST number digit"),
    ("KA51AB234",           "car",        "black",   "Hidden FIRST number digit"),
    ("KA05N9876",           "motorcycle", "red",     "Hidden SERIES letter (M)"),
    ("KA5MN9876",           "motorcycle", "red",     "Hidden DISTRICT digit"),
    ("MH12DE4567",          "suv",        "blue",    "Different state (MH), perfect"),
    ("KA51AB23",            "car",        "black",   "Hidden TWO number digits"),
    ("ZZ99QQ9999",          None,         None,      "Unknown / invalid plate"),
    ("KA",                  None,         None,      "Almost fully hidden -> unreliable"),
]


def main():
    path = os.path.join(tempfile.mkdtemp(), "demo.db")
    dbmod.build_from_records(path, FLEET)
    print("=" * 78)
    print(" ANPR with Partial Plate Recovery  --  inference engine demo")
    print("=" * 78)
    with PlateRecognizer(db_path=path, auto_build=False) as r:
        for ocr, vtype, color, label in SCENARIOS:
            res = r.identify(ocr, vehicle_type=vtype, vehicle_color=color, top_k=3)
            print(f"\n[{label}]")
            print(f"  OCR in : '{ocr}'   detected type={vtype} color={color}")
            print(f"  result : {res.summary()}")
            if res.reliable and res.ranked:
                print("  ranked :")
                for p in res.ranked:
                    db = "DB+" if p.in_database else "no "
                    print(f"     {db} {p.pretty:14s} {p.confidence:5.1f}%  "
                          f"{p.breakdown}")


if __name__ == "__main__":
    main()
