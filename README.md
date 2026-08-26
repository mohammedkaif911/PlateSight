# PlateSight

**Vehicle identification from partially hidden number plates.**

Standard ANPR systems return nothing when plate characters are missing. PlateSight predicts them — recovering the full plate number even when characters are physically hidden, blurred, or intentionally tampered with.

**Live demo:** https://platesight.onrender.com

## Results

**Overall recovery on hidden plates: 95.3%** (standard ANPR: 0%)

Benchmark: 50 plates × 7 hiding scenarios (350 trials), with realistic OCR noise simulated.

| Scenario | Hidden | Standard ANPR | PlateSight | Avg Confidence |
|---|---|---|---|---|
| Clear plate | 0 | 100% | 98.0% | 67.9% |
| 1 number digit | 1 | 0% | 90.0% | 65.3% |
| 2 number digits | 2 | 0% | 98.0% | 62.7% |
| 3 number digits | 3 | 0% | 84.0% | 60.0% |
| 1 series letter | 1 | 0% | 100% | 64.7% |
| 1 district digit | 1 | 0% | 100% | 65.1% |
| 1 state letter | 1 | 0% | 100% | 64.8% |

Plate detector: YOLOv8 fine-tuned on 1,936 images, mAP50 = 99.5%.

Unit tests: 17 passing.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## How It Works

**Stage 1 — Detection.** YOLOv8 finds the vehicle and number plate. Vehicle type and colour are extracted.

**Stage 2 — OCR.** fast-plate-ocr reads visible characters. Supports two-line Indian plates.

**Stage 3 — Recovery Engine.** The OCR string is aligned to the Indian plate format using dynamic programming. The alignment reveals which characters are hidden and which were misread. The engine generates candidates, filters them through the vehicle database, and ranks survivors with a transparent confidence score.

Two correction mechanisms handle OCR errors:

- **Confusion table** — maps visually similar pairs (Z/2, I/1, O/0, G/6, B/8) so misreads become recoverable substitutions
- **Database-guided relaxation** — if no match is found, relaxes each character one at a time to search for a database hit

Confidence scoring weights: OCR alignment quality (35%), database match (30%), vehicle model match (15%), vehicle type match (10%), valid state code (10%).

## Project Structure

```
anpr/
  plate_format.py         Indian plate format and state codes
  alignment.py            DP alignment for hidden/misread detection
  candidate_gen.py        Format-aware candidate generation
  database.py             Mock SQLite vehicle database
  scoring.py              Weighted confidence scoring
  engine.py               Orchestrator and DB-guided relaxation

server.py                 FastAPI backend
cv_pipeline.py            YOLOv8 + OCR front-end
static/index.html         Web interface
benchmark.py              Generates results table
tests/                    Unit tests
scripts/                  Helper scripts
```

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | /api/identify | Identify from plate string |
| POST | /api/upload | Identify from uploaded image |
| GET | /api/results | Benchmark results as JSON |
| POST | /api/add_vehicle | Add a vehicle to the database |

## Deploy

### Render (free)

1. Push this repo to GitHub
2. Create a Web Service on [render.com](https://render.com)
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn server:app --host 0.0.0.0 --port $PORT`

### Docker

```bash
docker build -t platesight .
docker run -p 8000:8000 platesight
```

## Data

All vehicle records are synthetic test data. The database auto-generates 500 random Indian plates on startup. No real vehicle or owner information is stored.

## Tech Stack

Python · FastAPI · YOLOv8 · fast-plate-ocr · OpenCV · SQLite · Docker
