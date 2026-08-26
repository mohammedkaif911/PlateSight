"""Indian standard number-plate format knowledge.

Standard format:  [STATE 2][DISTRICT 2][SERIES 1-3][NUMBER 4]
Example:          KA51AB1234  ->  KA 51 AB 1234

Documented simplifications (match real-world Indian plates):
  * DISTRICT is always 2 digits (all Indian RTO codes are 01-99).
  * NUMBER is displayed as 4 digits (standard, leading zeros shown).
  * SERIES varies 1-3 letters -> the only structural variability we model,
    so a "shape" is fully described by the series length.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Official Indian state / Union-Territory RTO codes.
VALID_STATES = {
    "AN": "Andaman & Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh", "CG": "Chhattisgarh",
    "DD": "Daman & Diu / DNH", "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat",
    "HR": "Haryana", "HP": "Himachal Pradesh", "JH": "Jharkhand", "JK": "Jammu & Kashmir",
    "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh", "LD": "Lakshadweep",
    "MH": "Maharashtra", "ML": "Meghalaya", "MN": "Manipur", "MP": "Madhya Pradesh",
    "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "PB": "Punjab",
    "PY": "Puducherry", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu",
    "TR": "Tripura", "TS": "Telangana", "UK": "Uttarakhand", "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

# Common RTO district names (partial lookup)
DISTRICT_NAMES = {
    "KA01": "Bengaluru East", "KA02": "Bengaluru West", "KA03": "Bengaluru North",
    "KA04": "Bengaluru Rural", "KA05": "Bengaluru South", "KA06": "Devanahalli",
    "KA07": "Bengaluru Central", "KA09": "Mysuru", "KA10": "Davanagere",
    "KA11": "Tumakuru", "KA12": "Kolar", "KA14": "Tumakuru Rural",
    "KA15": "Belagavi", "KA17": "Hubballi", "KA18": "Dharwad",
    "KA20": "Bagalkote", "KA21": "Bengaluru Yeshwanthpur", "KA22": "Kalaburagi",
    "KA23": "Ballari", "KA24": "Bidar", "KA25": "Raichur", "KA26": "Koppal",
    "KA28": "Chitradurga", "KA30": "Shivamogga", "KA31": "Hassan",
    "KA32": "Dakshina Kannada", "KA33": "Udupi", "KA34": "Chikkamagaluru",
    "KA36": "Mandya", "KA41": "Vijayapura", "KA50": "Bengaluru Electronic City",
    "KA51": "Bengaluru Yeshwanthpur", "KA52": "Bengaluru Nelamangala",
    "KA53": "Bengaluru Ramanagara", "KA54": "Bengaluru Kengeri",
    "KA55": "Mangaluru",
    "MH01": "Mumbai Central", "MH02": "Mumbai West", "MH12": "Pune", "MH14": "Pimpri-Chinchwad",
    "DL01": "Delhi North", "DL02": "Delhi South", "DL03": "Delhi West", "DL10": "Delhi North-West",
    "TN01": "Chennai Central", "TN02": "Chennai South", "TN07": "Chennai West", "TN22": "Coimbatore",
    "TS09": "Hyderabad", "AP09": "Hyderabad", "AP28": "Visakhapatnam",
    "KL01": "Thiruvananthapuram", "KL07": "Kochi",
    "GJ01": "Ahmedabad", "GJ27": "Surat",
    "RJ01": "Jaipur", "RJ14": "Jodhpur",
    "UP16": "Kanpur", "UP32": "Lucknow", "UP70": "Noida",
    "WB01": "Kolkata North", "WB02": "Kolkata South", "WB20": "Howrah",
    "BR01": "Patna", "BR02": "Gaya",
}


def decode_plate(plate_str):
    """Return decoded info: state_code, state_name, district_code, district_name."""
    plate_str = normalize(plate_str)
    if len(plate_str) < 4:
        return None
    state = plate_str[:2]
    district = plate_str[2:4]
    state_name = VALID_STATES.get(state, "Unknown")
    key = f"{state}{district}"
    district_name = DISTRICT_NAMES.get(key, f"RTO Code {district}")
    return {
        "state_code": state,
        "state_name": state_name,
        "district_code": district,
        "district_name": district_name,
    }

CLASS_LETTER, CLASS_DIGIT = "L", "D"


@dataclass(frozen=True)
class PlateShape:
    """A plate layout. Only the series length varies."""
    series_n: int = 2
    district_n: int = 2
    number_n: int = 4
    state_n: int = 2

    @property
    def total(self) -> int:
        return self.state_n + self.district_n + self.series_n + self.number_n

    def klasses(self) -> str:
        """Character class per cell, e.g. 'LLDDLLDDDD'."""
        return (
            CLASS_LETTER * self.state_n
            + CLASS_DIGIT * self.district_n
            + CLASS_LETTER * self.series_n
            + CLASS_DIGIT * self.number_n
        )

    def segment_bounds(self) -> List[Tuple[int, int]]:
        """(start, end) index ranges for [state, district, series, number]."""
        bounds, i = [], 0
        for length in (self.state_n, self.district_n, self.series_n, self.number_n):
            bounds.append((i, i + length))
            i += length
        return bounds


def all_shapes() -> List[PlateShape]:
    """All plausible plate shapes (series length 1-3)."""
    return [PlateShape(series_n=s) for s in range(1, 4)]


def normalize(ocr_text: str) -> str:
    """Uppercase and keep only A-Z0-9 (drops spaces, dashes, dirt, etc.)."""
    return "".join(c for c in ocr_text.upper() if c.isalnum())


def split(plate_str: str, shape: PlateShape) -> Tuple[str, str, str, str]:
    (ss, se), (ds, de), (ls, le), (ns, ne) = shape.segment_bounds()
    return plate_str[ss:se], plate_str[ds:de], plate_str[ls:le], plate_str[ns:ne]


def pretty(plate_str: str, shape: PlateShape) -> str:
    """Human-readable spacing: KA51AB1234 -> KA 51 AB 1234."""
    st, di, se, nu = split(plate_str, shape)
    return f"{st} {di} {se} {nu}"


def parse(plate_str: str) -> Optional[Tuple[str, str, str, str, PlateShape]]:
    """Parse a (lenient) plate string into segments + shape, or None."""
    plate_str = normalize(plate_str)
    m = re.fullmatch(r"([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})", plate_str)
    if not m:
        return None
    state, district, series, number = m.groups()
    return state, district, series, number, PlateShape(series_n=len(series))
