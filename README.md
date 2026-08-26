# PlateSight

### **Vehicle identification from partially hidden, blurred, or tampered number plates.**

> **Live demo:** [platesight.onrender.com](https://platesight.onrender.com)

Standard ANPR systems **fail completely** when even a single character is missing from a number plate. PlateSight **predicts the missing characters**, generates candidates, filters them through a vehicle database, and returns the most probable match with a **transparent confidence score**.

**95.3% recovery rate** on hidden plates vs **0%** for traditional ANPR.

---

## Overview

In real-world traffic scenarios, vehicle owners frequently hide number plate digits using stickers, mud, scratches, or physical modifications to avoid identification during violations. Environmental factors like poor lighting, rain, motion blur, and low-resolution cameras further degrade plate readability.

Existing ANPR systems depend on **clearly visible** plates and cannot handle incomplete information. **PlateSight** fills this gap by combining:

- **Computer vision** (YOLOv8) for vehicle and plate detection
- **OCR** (fast-plate-ocr) for reading visible characters
- **A recovery engine** that predicts missing characters using template alignment, candidate generation, and database-guided filtering
- **A confusion table** and **DB-guided relaxation** to correct OCR misreads

---

## Key Features

- **Hidden character recovery** — Aligns the OCR read to the Indian plate format and identifies exactly which characters are missing
- **OCR error correction** — A confusion table maps visually similar pairs (Z↔2, I↔1, O↔0, G↔6, B↔8) so misreads become recoverable
- **Database-guided relaxation** — When the initial pass finds no match, the engine relaxes each character one at a time (and pairs within the same segment) to find a database hit
- **Two-line plate support** — Splits the crop into top/bottom halves for standard Indian plates where state+district sit above series+number
- **Attribute-based filtering** — Candidates are filtered by **vehicle type**, **color**, and **model** before database matching
- **Transparent scoring** — Every prediction includes a full breakdown showing how the confidence percentage was calculated
- **Dark / light themes** with a fully responsive layout

---

## How It Works

### Stage 1 — Detection

**YOLOv8** (COCO-pretrained) detects the vehicle and classifies it as car, motorcycle, bus, or truck. A separate **plate detector** (fine-tuned YOLOv8, 99.5% mAP) localizes the number plate region. Vehicle **type** and **dominant color** are extracted for downstream filtering.

### Stage 2 — OCR

**fast-plate-ocr** (plate-specialised ONNX model) reads the visible characters from the plate crop. For **two-line Indian plates**, the crop is split horizontally and the top read (state+district) is concatenated with the bottom read (series+number).

### Stage 3 — Recovery Engine (Core Contribution)

The OCR string is aligned to the Indian plate format `[STATE][DISTRICT][SERIES][NUMBER]` using a **dynamic-programming alignment**. This alignment classifies each position as:

- **Matched** — OCR read this character correctly
- **Soft** — OCR read a visually similar but wrong character (confusable)
- **Hidden** — The character is missing from the OCR read entirely

The engine then **generates candidates** for hidden and soft positions, **filters** them through the vehicle database using attributes, and **ranks** survivors with a weighted score:

| Signal | Weight |
|---|---|
| OCR alignment quality | **35%** |
| Database match | **30%** |
| Vehicle model match | **15%** |
| Vehicle type match | **10%** |
| Valid state code | **10%** |

#### OCR Error Correction

Two mechanisms handle misreads:

> **Confusion Table** — Visually similar pairs (Z↔2, I↔1, O↔0, G↔6, B↔8) are mapped so a misread becomes a *soft substitution* with a tiny candidate set (1-3 options) instead of a full alphabet search.

> **Database-Guided Relaxation** — If the straight pass finds no database match, the engine *relaxes* each confident character one at a time (treating it as hidden) to search for a match. For harder cases, it relaxes **pairs of characters** within the same segment. This recovers misreads like **0→8** that no confusion table can catch.

---

## Benchmark Results

**50 plates** × **7 hiding scenarios** = **350 trials**, with realistic OCR noise (12% per-character misread rate).

### Overall: **95.3% recovery** on hidden plates (Standard ANPR: **0%**)

| Scenario | Hidden | Standard ANPR | PlateSight | Avg Confidence |
|---|---|---|---|---|
| Clear plate | 0 | 100% | 98.0% | 67.9% |
| 1 number digit | 1 | 0% | 90.0% | 65.3% |
| 2 number digits | 2 | 0% | 98.0% | 62.7% |
| 3 number digits | 3 | 0% | 84.0% | 60.0% |
| 1 series letter | 1 | 0% | 100% | 64.7% |
| 1 district digit | 1 | 0% | 100% | 65.1% |
| 1 state letter | 1 | 0% | 100% | 64.8% |

> An earlier benchmark (pre-relaxation engine) reported 86.3%. The improvement to **95.3%** comes from the two-cell DB-guided relaxation fix.

**Plate detector:** YOLOv8 fine-tuned on 1,936 images (train 1596 / val 226 / test 114) — **mAP50: 99.5%**, precision: 99.5%, recall: 100%.

**Unit tests:** 17/17 passing.

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/yourusername/platesight.git
cd platesight
pip install -r requirements.txt
```

### Running the server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

### Running tests

```bash
python tests/test_engine.py
```

### Generating the benchmark

```bash
python benchmark.py
```

This writes `results.md`, `results.csv`, and `results.json`.

---

## Usage

### Manual Entry

Type the characters OCR managed to read. Leave out any character to mark it as **hidden**. The engine will attempt to recover it.

**Example:** Enter `KA21NJY9` (3 trailing digits hidden) → recovers `KA 21 NJY 9125` at **91% confidence**.

### Image Upload

Upload a clear photo of a vehicle. The system will:
1. **Detect** the vehicle and plate
2. **Read** the visible characters via OCR
3. **Recover** any hidden characters
4. **Match** against the vehicle database
5. Display the prediction with a full **confidence breakdown**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/identify` | Identify from a plate string (manual entry) |
| `POST` | `/api/upload` | Identify from an uploaded image |
| `GET` | `/api/results` | Get benchmark results as JSON |
| `POST` | `/api/add_vehicle` | Add a vehicle to the database |
| `GET` | `/api/db_count` | Get total vehicle count |

**Example request:**

```bash
curl -X POST http://localhost:8000/api/identify \
  -H "Content-Type: application/json" \
  -d '{"plate": "KA21NJY9", "type": "car", "color": "white"}'
```

---

## Deployment

### Deploy to Render (free)

1. **Push** this repository to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repository
4. Set:
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Click **Create** — Render handles the rest

> Render's free tier spins down after 15 min of inactivity. First request after spin-down takes ~30s.

### Deploy with Docker

```bash
docker build -t platesight .
docker run -p 8000:8000 platesight
```

The plate detector weights are in `models/`. YOLOv8 weights and the OCR model **auto-download** from Hugging Face on the first API call.

---

## Project Structure

```
anpr/                         Recovery engine (core contribution)
  plate_format.py             Indian plate format, state codes, shape model
  alignment.py                DP alignment — detects hidden + misread chars
  candidate_gen.py            Format-aware candidate generation
  database.py                 Mock SQLite vehicle database
  scoring.py                  Weighted confidence scoring
  engine.py                   Orchestrator + DB-guided relaxation

cv_pipeline.py                YOLOv8 + plate OCR front-end
server.py                     FastAPI backend (UI + JSON API)
static/index.html             Web interface (dark/light, responsive)
benchmark.py                  Generates the results table
train.py                      Plate detector training (Google Colab)
tests/                        17 unit tests
scripts/                      Helper scripts
```

---

## Training the Plate Detector

The plate detector was fine-tuned on the **Roboflow "plate-number"** dataset (1,936 images) using **YOLOv8** in Google Colab.

```bash
# Run in Google Colab (GPU runtime)
!pip install -q ultralytics roboflow
!python train.py --api-key YOUR_KEY --workspace fyp-hq4ka \
  --project plate-number-dwkyk --version 1 --epochs 60
```

**Results:** mAP50 = 99.5%, precision = 99.5%, recall = 100%

See `TRAINING.md` for the full step-by-step guide.

---

## Data

> All vehicle records in the database are **synthetic test data**. The mock database generates 500 random Indian plates on startup. No real vehicle or owner information is stored. Test plates can be added through the web interface.

---

## Tech Stack

| Category | Technology |
|---|---|
| **Backend** | Python, FastAPI, Uvicorn |
| **Detection** | YOLOv8 (Ultralytics) |
| **OCR** | fast-plate-ocr (ONNX) |
| **Image Processing** | OpenCV |
| **Database** | SQLite |
| **Frontend** | HTML, CSS, JavaScript |
| **Deployment** | Docker, Render |

---

## License

This project is developed as an **academic major project**. All vehicle data is synthetic.
