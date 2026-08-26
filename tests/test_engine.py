"""Test suite for the ANPR inference engine.

Run with:  python tests/test_engine.py
Also pytest-compatible (functions named test_*).
"""
import os
import sys
import tempfile

# Make the package importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anpr import PlateRecognizer, normalize  # noqa: E402
from anpr import database as dbmod            # noqa: E402
from anpr.alignment import align              # noqa: E402

# A fixed, known vehicle set so scenarios are deterministic.
KNOWN_RECORDS = [
    {"plate": "KA51AB1234", "vehicle_type": "car",       "color": "black",  "brand": "Maruti Suzuki"},
    {"plate": "KA05MN9876", "vehicle_type": "motorcycle","color": "red",    "brand": "Bajaj"},
    {"plate": "KA53XY0001", "vehicle_type": "car",       "color": "white",  "brand": "Hyundai"},
    {"plate": "MH12DE4567", "vehicle_type": "suv",       "color": "blue",   "brand": "Mahindra"},
    {"plate": "TN07FG8910", "vehicle_type": "car",       "color": "silver", "brand": "Honda"},
]


def _recognizer():
    path = os.path.join(tempfile.mkdtemp(), "test.db")
    dbmod.build_from_records(path, KNOWN_RECORDS)
    return PlateRecognizer(db_path=path, auto_build=False)


# ---- format / alignment unit checks ----------------------------------------
def test_normalize():
    assert normalize("ka 51-ab.1234") == "KA51AB1234"
    assert normalize(" KA05\tMN 9876 ") == "KA05MN9876"


def test_align_perfect_read():
    a = align("KA51AB1234")
    assert a.missing_count == 0
    assert a.pattern_string() == "KA51AB1234"


def test_align_detects_hidden_number_digit():
    # last number digit hidden
    a = align("KA51AB123")
    assert a.missing_count == 1
    # the single '?' must be inside the number segment (positions 6..9)
    assert a.missing_positions[0] >= 6


# ---- end-to-end engine scenarios -------------------------------------------
def test_perfect_read_predicts_known_plate():
    with _recognizer() as r:
        res = r.identify("KA51AB1234", vehicle_type="car", vehicle_color="black")
    assert res.reliable
    assert res.best.plate == "KA51AB1234"
    assert res.best.in_database
    assert res.best.confidence >= 80.0


def test_hidden_last_number_digit():
    with _recognizer() as r:
        res = r.identify("KA51AB123", vehicle_type="car", vehicle_color="black")
    assert res.reliable
    assert res.best.plate == "KA51AB1234"
    assert res.best.in_database
    assert res.best.confidence >= 70.0


def test_hidden_first_number_digit_recovered_via_subsequence():
    # '1' hidden from 1234 -> OCR sees "234"; subsequence placement must recover 1234
    with _recognizer() as r:
        res = r.identify("KA51AB234", vehicle_type="car", vehicle_color="black")
    assert res.reliable
    assert res.best.plate == "KA51AB1234"


def test_hidden_series_letter():
    # KA05MN9876 with 'M' hidden -> OCR sees series "N"
    with _recognizer() as r:
        res = r.identify("KA05N9876", vehicle_type="motorcycle", vehicle_color="red")
    assert res.reliable
    assert res.best.plate == "KA05MN9876"


def test_hidden_district_digit():
    # KA05MN9876 with one district digit hidden -> OCR "KA5MN9876"
    with _recognizer() as r:
        res = r.identify("KA5MN9876", vehicle_type="motorcycle", vehicle_color="red")
    assert res.reliable
    assert res.best.plate == "KA05MN9876"


def test_unknown_plate_low_confidence():
    with _recognizer() as r:
        res = r.identify("ZZ99QQ9999")  # invalid state, not in DB
    assert res.reliable
    assert not res.best.in_database
    assert res.best.confidence < 50.0


def test_too_many_hidden_is_unreliable():
    with _recognizer() as r:
        res = r.identify("KA")  # almost everything hidden
    assert not res.reliable
    assert res.missing_count > 2


def test_ranked_returns_top_k_and_sorted():
    with _recognizer() as r:
        res = r.identify("KA51AB123", top_k=3)
    confs = [p.confidence for p in res.ranked]
    assert confs == sorted(confs, reverse=True)
    assert len(res.ranked) <= 3


# ---- misread (confusable) handling -----------------------------------------
def test_confusable_targets():
    from anpr.alignment import confusable_targets
    assert confusable_targets("Z", "D") == {"2"}
    assert confusable_targets("I", "D") == {"1"}
    assert confusable_targets("G", "D") == {"6", "9"}
    assert confusable_targets("O", "D") == {"0"}
    assert confusable_targets("A", "D") == set()  # not confusable


def test_misread_digit_is_recovered():
    # OCR misreads the number '1' as letter 'I' -> must still recover KA51AB1234
    with _recognizer() as r:
        res = r.identify("KA51ABI234", vehicle_type="car", vehicle_color="black")
    assert res.reliable
    assert res.best.plate == "KA51AB1234"
    assert res.best.in_database


def test_misread_zero_as_O_recovered():
    # need a plate with a 0 in a confusable spot
    import os as _os, tempfile as _tf
    from anpr import database as _db
    path = _os.path.join(_tf.mkdtemp(), "t.db")
    _db.build_from_records(path, KNOWN_RECORDS + [
        {"plate": "KA51AB0234", "vehicle_type": "car", "color": "black", "brand": "X"}])
    with PlateRecognizer(db_path=path, auto_build=False) as r:
        res = r.identify("KA51ABO234")  # '0' read as 'O'
    assert res.best.plate == "KA51AB0234"


# ---- runner ----------------------------------------------------------------
def test_hidden_state_letter_recovered():
    # hide one state letter (the 'A' in KA) -> only 'K' visible
    with _recognizer() as r:
        res = r.identify("K51AB1234", vehicle_type="car", vehicle_color="black")
    assert res.reliable
    assert res.best.plate == "KA51AB1234"


def test_relaxation_recovers_digit_misread():
    # OCR misreads 0 as 8 (digit->digit) in a FULL read -> only relaxation can fix it
    import os as _os, tempfile as _tf
    from anpr import database as _db
    path = _os.path.join(_tf.mkdtemp(), "t.db")
    _db.build_from_records(path, [
        {"plate": "TN02AK8055", "vehicle_type": "car", "color": "black",
         "brand": "BMW", "model": "320b"}])
    with PlateRecognizer(db_path=path, auto_build=False) as r:
        res = r.identify("TN82AK8055")  # 0 misread as 8
    assert res.best.plate == "TN02AK8055"
    assert res.via_relaxation


def test_relaxation_two_misreads():
    # two number digits misread ('12' -> '99'): needs two-cell relaxation
    import os as _os, tempfile as _tf
    from anpr import database as _db
    path = _os.path.join(_tf.mkdtemp(), "t.db")
    _db.build_from_records(path, [
        {"plate": "KA51AB1234", "vehicle_type": "car", "color": "white",
         "brand": "Honda", "model": "City"}])
    with PlateRecognizer(db_path=path, auto_build=False) as r:
        res = r.identify("KA51AB9934")  # '12' misread as '99'
    assert res.best.plate == "KA51AB1234"
    assert res.via_relaxation


# ---- runner ----------------------------------------------------------------
def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            raise
    print(f"\n{passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    _run()
