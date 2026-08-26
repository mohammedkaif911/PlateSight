"""Mock vehicle database (SQLite).

Since official RTO access is restricted, we generate a realistic mock database
for academic / testing purposes. Each record: plate, vehicle_type, color,
brand, model.
"""
from __future__ import annotations

import os
import random
import sqlite3
from typing import Dict, Iterable, List, Optional

from .plate_format import VALID_STATES

VEHICLE_TYPES = ["car", "suv", "motorcycle", "auto", "bus", "truck"]
COLORS = ["white", "black", "red", "blue", "silver", "grey", "green", "yellow", "brown"]
SERIES_LETTERS = "ABCDEFGHJKLMNPRSTVWXYZ"  # I/O/Q/U avoided

BRANDS_CAR = ["Maruti Suzuki", "Hyundai", "Honda", "Toyota", "Tata",
              "Mahindra", "Kia", "Volkswagen", "Ford", "Renault"]
BRANDS_BIKE = ["Hero", "Bajaj", "TVS", "Royal Enfield", "Yamaha", "Honda", "KTM", "Suzuki"]

FIRST_NAMES = ["Rajesh", "Suresh", "Priya", "Arun", "Lakshmi", "Mohammed", "Deepa",
               "Vinod", "Sunita", "Anil", "Fatima", "Ravi", "Geeta", "Imran",
               "Naveen", "Shweta", "Karthik", "Anjali", "Faisal", "Divya"]
LAST_NAMES = ["Kumar", "Sharma", "Reddy", "Nair", "Patel", "Gowda", "Khan", "Singh",
              "Rao", "Iyer", "Das", "Sheikh", "Pillai", "Shetty", "Menon"]

MODELS = {
    "Maruti Suzuki": ["Swift", "Baleno", "Brezza", "Dzire", "Wagon R"],
    "Hyundai": ["i20", "Creta", "Verna", "Venue", "Grand i10"],
    "Honda": ["City", "Amaze", "WR-V", "Jazz"],
    "Toyota": ["Innova", "Urban Cruiser", "Glanza", "Fortuner"],
    "Tata": ["Nexon", "Harrier", "Tiago", "Altroz", "Punch"],
    "Mahindra": ["XUV700", "Scorpio", "Thar", "Bolero", "XUV300"],
    "Kia": ["Seltos", "Sonet", "Carens"],
    "Volkswagen": ["Polo", "Virtus", "Taigun"],
    "Ford": ["EcoSport", "Endeavour"],
    "Renault": ["Kwid", "Triber", "Kiger"],
    "Hero": ["Splendor", "HF Deluxe", "Passion", "Glamour"],
    "Bajaj": ["Pulsar", "Platina", "Dominar", "CT"],
    "TVS": ["Apache", "Jupiter", "Ntorq", "Radeon"],
    "Royal Enfield": ["Classic 350", "Bullet 350", "Hunter 350", "Meteor"],
    "Yamaha": ["FZ", "MT-15", "R15", "RayZR"],
    "KTM": ["Duke 200", "Duke 250", "RC 200"],
    "Suzuki": ["Gixxer", "Access", "Burgman"],
}


def _rand_plate(rng: random.Random) -> str:
    state = rng.choice(list(VALID_STATES))
    district = f"{rng.randint(1, 99):02d}"
    series = "".join(rng.choice(SERIES_LETTERS) for _ in range(rng.choice([2, 2, 2, 3])))
    number = f"{rng.randint(0, 9999):04d}"
    return state + district + series + number


def _create_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute("DROP TABLE IF EXISTS vehicles")
    cursor.execute(
        "CREATE TABLE vehicles ("
        "plate TEXT PRIMARY KEY, vehicle_type TEXT, color TEXT, brand TEXT, model TEXT, "
        "owner_name TEXT, owner_age INTEGER, owner_address TEXT)"
    )


def build_database(path: str, n: int = 500, seed: int = 42) -> str:
    """Create a fresh random mock database with n unique vehicles.

    A separate RNG is used for model selection so the generated plates (which
    the test images depend on) stay identical for a given seed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        _create_table(cur)
        rng = random.Random(seed)
        rng_model = random.Random(seed + 1)
        seen, rows = set(), []
        CITIES = ["Bengaluru", "Mysuru", "Mumbai", "Pune", "Delhi", "Chennai",
                  "Hyderabad", "Kochi", "Jaipur", "Kolkata", "Ahmedabad", "Patna"]
        while len(rows) < n:
            plate = _rand_plate(rng)
            if plate in seen:
                continue
            seen.add(plate)
            vtype = rng.choice(VEHICLE_TYPES)
            brand = rng.choice(BRANDS_BIKE if vtype == "motorcycle" else BRANDS_CAR)
            color = rng.choice(COLORS)
            model = rng_model.choice(MODELS.get(brand, ["Standard"]))
            owner = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            age = rng.randint(21, 65)
            addr = f"{rng.randint(1, 999)}, {rng.choice(['MG Road', 'Brigade Road', 'JP Nagar', 'Indiranagar', 'Koramangala', 'HSR Layout'])}, {rng.choice(CITIES)}"
            rows.append((plate, vtype, color, brand, model, owner, age, addr))
        cur.executemany(
            "INSERT INTO vehicles VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    return path


def build_from_records(path: str, records: Iterable[Dict]) -> str:
    """Create a database from explicit records."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        _create_table(cur)
        rows = [(r["plate"], r.get("vehicle_type", ""), r.get("color", ""),
                 r.get("brand", ""), r.get("model", ""),
                 r.get("owner_name", ""), r.get("owner_age", ""),
                 r.get("owner_address", "")) for r in records]
        cur.executemany(
            "INSERT INTO vehicles VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    return path


def add_record(conn: sqlite3.Connection, record: Dict) -> bool:
    """Insert/replace a single record."""
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO vehicles "
        "(plate, vehicle_type, color, brand, model, owner_name, owner_age, owner_address) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (record["plate"], record.get("vehicle_type", ""), record.get("color", ""),
         record.get("brand", ""), record.get("model", ""),
         record.get("owner_name", ""), record.get("owner_age", ""),
         record.get("owner_address", "")),
    )
    conn.commit()
    return True


def query(conn: sqlite3.Connection, plate: str) -> Optional[Dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT plate, vehicle_type, color, brand, model, owner_name, owner_age, owner_address "
        "FROM vehicles WHERE plate = ?",
        (plate,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {"plate": row[0], "vehicle_type": row[1], "color": row[2],
            "brand": row[3], "model": row[4], "owner_name": row[5],
            "owner_age": row[6], "owner_address": row[7]}
