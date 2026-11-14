# AI-Based Incident-Time Person Reconstruction in an Unobservable Path

An academic, end-to-end application that estimates **which people were most
likely inside an unobservable (camera-free) path at the exact time of an
incident**, using CCTV observations before and after the incident, a connected
path network, and travel-time characteristics.

## Central research question

> Given a person's movement timeline, path connectivity, travel-time
> characteristics, and observations before and after an incident, what is the
> probability that the person was inside the affected unobservable path at the
> exact incident timestamp?

## Technology stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Person detection | YOLO (Ultralytics) |
| Tracking | ByteTrack |
| Video | OpenCV |
| Data | Pandas, NumPy |
| Path network | NetworkX |
| ML | Scikit-learn (only when labeled data exists) |
| Database | MongoDB (PyMongo) |
| Env | python-dotenv |
| Dashboard | Streamlit |
| Charts | Plotly |

Strictly avoided: Docker, Kubernetes, React, Node, FastAPI, WebSockets,
GNN/LSTM/GRU, facial recognition, cloud services, auth systems, GPU requirements.

## Project structure

```
incident-reconstruction/
app.py                      # Streamlit dashboard (10 pages)
modules/
  video_processor.py        # OpenCV video processing -> observations
  person_detector.py        # YOLO person detection
  person_tracker.py         # ByteTrack tracking
  timeline_generator.py     # Per-person timelines (observed vs inferred)
  path_manager.py           # NetworkX path graph + CRUD
  travel_time.py            # Travel-time estimation
  reconstruction_engine.py  # Hidden-path probability engine (core)
  threat_engine.py          # HIGH/MEDIUM/LOW classification
  post_incident_tracker.py  # Post-incident movement
  evaluation.py             # Metrics vs ground truth
database/
  connection.py             # MongoDB connection (env-driven)
  db_operations.py          # CRUD + indexes
  schema.py                 # Collections + index definitions
models/
  model_training.py         # Placeholder (no training without labeled data)
data/
  synthetic_data.py         # Demo dataset generator (no video needed)
utils/
  config.py                 # Config + thresholds
  helpers.py
tests/                      # pytest unit tests
reports/                    # CSV exports destination
.env.example                # Copy to .env and edit
requirements.txt
```

## Installation

### 1. Install Python
Python 3.11 or later. From this directory:

```bash
pip install -r requirements.txt
```

### 2. Install MongoDB locally
Install MongoDB Community Server (Windows). The server binary is expected at:

```
C:\Program Files\MongoDB\Server\8.0\bin\mongod.exe
```

Start it with a data directory, e.g. in PowerShell:

```powershell
New-Item -ItemType Directory -Force C:\data\db
"C:\Program Files\MongoDB\Server\8.0\bin\mongod.exe" --dbpath C:\data\db
```

Or use MongoDB Atlas and set `MONGODB_URI` in `.env`.

### 3. Configure environment
Copy `.env.example` to `.env`:

```
MONGODB_URI=mongodb://localhost:27017/
DATABASE_NAME=incident_reconstruction
```

No credentials are hardcoded anywhere.

### 4. YOLO model weights
Ultralytics downloads `yolov8n.pt` automatically on first run (CPU inference;
no GPU required).

### 5. Run the application
```bash
streamlit run app.py
```

## Quick demo (no video needed)

1. Start the app: `streamlit run app.py`
2. In the sidebar, click **Generate demo dataset** — this creates paths P1–P6
   (P4 unobservable), cameras, 20 people, camera observations and an incident
   (INC_001 on P4 at 10:30), with ground truth stored separately.
3. Click **Run demo incident** — the reconstruction engine evaluates every
   person and classifies them HIGH / MEDIUM / LOW.
4. Open **Affected-Person Analysis** to see the ranked list, or
   **Evaluation** to compare predictions against ground truth
   (precision, recall, F1, localization accuracy, path reconstruction accuracy,
   travel-time error).

## How the core algorithm works

For every person, the engine:

1. Loads observations before and after the incident.
2. Finds feasible routes through the NetworkX path graph between those
   observations that pass through the affected path.
3. Builds a plausible presence window on the affected path using min/expected/
   max travel times (path length / walking speed).
4. Computes a triangular temporal-membership score at the incident timestamp,
   scaled by route coverage, observation recency and the person's actual
   historical transitions through the path.
5. Outputs a probability in [0, 100], an estimated entry/exit time, a
   reconstructed route and a text explanation.

Physically impossible scenarios (no feasible route, insufficient time) yield
probability 0. No random probabilities are ever produced, and no facial
recognition or person re-identification is used — cross-camera identity is
associated manually.

## Evaluation

The `ground_truth` collection (demo only) records who was actually on the
affected path. Metrics are computed against it, never against the system's own
predictions:

- Affected-person precision / recall / F1
- Incident-time localization accuracy
- Path reconstruction accuracy
- Travel-time estimation error

## Running tests

```bash
pip install pytest
pytest tests/
```

Tests require a running local MongoDB instance.

## License

Academic project. Probability estimates indicate potential exposure to an
incident — not guilt or wrongdoing.