# Reality Shield (Multimodel Deepfake Detector)

Reality Shield is a **FastAPI + React** web app for checking whether uploaded **images**, **videos**, or **audio** clips look **REAL** or **FAKE** (AI-generated or manipulated). Analysis runs **locally** on your machine: the backend compares uploads to reference samples in `data/raw/` and returns a verdict, confidence score, and detailed findings.

## What This Project Does

| Modality | How it is analyzed |
|----------|-------------------|
| **Image** | Visual features (color, texture, gradients) matched against real/fake reference images |
| **Video** | Up to 24 frames sampled per clip, per-frame matching, whole-clip similarity, and temporal stability cues |
| **Audio** | Librosa spectral/MFCC features matched against real/fake reference audio |

The UI shows scan history, analytics, and (for video) a per-frame breakdown with thumbnails.

## Recent Updates (Video Analysis)

Earlier versions often returned **INCONCLUSIVE** for videos because verdict logic used a wide “uncertain” band (roughly 42%–58% fake score).

**Current behavior:**

- **Whole-clip classification** — The full video feature vector is classified the same way as images and audio.
- **Smarter frame voting** — Up to **24 frames** per video; frame labels (REAL/FAKE) drive the vote, not only raw probabilities.
- **Merged scoring** — Combines video-level similarity, weighted frame scores, median frame score, vote ratio, and distance to real/fake clusters.
- **Temporal cues** — Brightness flicker, sharpness instability, and motion inconsistency slightly influence the fake score.
- **Clear REAL/FAKE results** — Borderline uploads resolve to REAL or FAKE using the reference classifier. **INCONCLUSIVE** is reserved for strong disagreement between whole-clip and frame-level signals, or when no frames can be read.

**For best video accuracy:** add quality reference videos under `data/raw/videos/` (see below) and upload short clips with a visible face, stable camera, and moderate compression.

---

## Prerequisites

Install on any new machine before cloning:

| Tool | Version |
|------|---------|
| **Git** | Latest stable |
| **Python** | 3.10 or newer (3.11+ recommended) |
| **Node.js** | 18 or newer with **npm** |

Optional (for training the full ML pipeline only):

- `requirements.txt` + `requirements-ml.txt` at the project root
- Enough disk space for datasets and `models/saved_models/`

---

## Set Up On Another Local Machine (Step by Step)

Follow these steps when you clone the repo on a **new PC**, laptop, or teammate’s machine.

### 1. Clone the repository

```powershell
git clone https://github.com/Vaishnavi-Chougale/RealityShield-AI-Generated-Content-Detector.git
cd RealityShield - AI Generated Content Detector
```

If you use a different remote or folder name (for example `Multimodel-Deepkafe-main`), `cd` into that folder instead. All commands below assume you are in the **project root** (where `backend/` and `frontend/` live).

### 2. Create a Python virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install backend dependencies

```powershell
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

This installs FastAPI, OpenCV, NumPy, Pillow, Librosa, PyTorch, and other packages used for local inference.

### 4. Reference data (included in the repository)

This repo ships with sample reference media under `data/raw/` and processed outputs under `data/processed/`. After you clone, these folders are already present—no manual copy step is required for the default dataset.

```text
data/
  raw/
    images/     real_*.jpg, fake_*.jpg
    videos/     real_*.mp4, fake_*.mp4
    audios/     real_*.wav, fake_*.wav
  processed/    preprocessed samples and CSV feature files
  metadata.csv
  sample_data.csv
```

**Naming rules** (when you add your own files):

- Files must start with `real_` or `fake_`.
- Include **both** real and fake examples in each folder; the backend builds centroids from these labels.

If you replace the dataset, keep the same naming convention. Without `data/raw/`, analysis will fail with a “reference data not found” error.

### 5. Optional environment file

Create `.env` in the project root only if you use optional cloud/LLM integrations:

```env
EMERGENT_LLM_KEY=your_api_key_here
```

Do **not** commit real API keys. Local image/video/audio scanning works without this key.

### 6. Start the backend (Terminal 1)

From the project root, with the virtual environment activated:

```powershell
python backend/run_local.py
```

The backend:

- Listens on `http://127.0.0.1:8000` when that port is free
- Picks the next free port if `8000` is busy
- Writes `frontend/.env.local` with `VITE_BACKEND_URL=http://127.0.0.1:<port>`

Leave this terminal open.

### 7. Install and start the frontend (Terminal 2)

```powershell
cd frontend
npm install
npm run dev
```

**macOS / Linux:** same commands from the `frontend` directory.

Open the URL Vite prints, usually:

```text
http://localhost:5173
```

### 8. Verify the stack

1. Backend health: open `http://127.0.0.1:8000/api/health` in a browser (use your actual port if different).
2. In the web UI: choose **Image**, **Video**, or **Voice**, upload a file, and run a scan.
3. Check **REAL** or **FAKE** with a confidence percentage and details. Video scans also show per-frame results.

---

## How To Use The Web App (Daily Workflow)

1. **Start backend first**, then **frontend** (so `.env.local` points to the correct API).
2. Open `http://localhost:5173`.
3. Select a media type tab: **Image**, **Video**, or **Voice**.
4. Click upload and choose a file:
   - Images: JPEG, PNG, WEBP
   - Video: MP4 and other formats OpenCV can read
   - Audio: WAV, MP3, etc. (Librosa; WAV is most reliable)
5. Submit the scan and read the result card:
   - **Verdict:** REAL, FAKE, or INCONCLUSIVE (rare for video after the latest logic)
   - **Confidence** and **risk level**
   - **Artifacts** and recommendations
   - **Video:** frame-by-frame thumbnails and labels
6. Recent scans appear in history/analytics until you **restart the backend** (history is in-memory only).

---

## Run Modes (Quick Reference)

### Full web app (recommended)

| Terminal | Command |
|----------|---------|
| 1 | `python backend/run_local.py` (from project root, venv active) |
| 2 | `cd frontend` → `npm install` → `npm run dev` |

### Frontend only

```powershell
cd frontend
npm install
npm run dev
```

The UI loads, but uploads fail until the backend is running.

### ML training pipeline (optional)

Trains/evaluates models and writes reports; not required for the upload UI.

```powershell
python -m pip install -r requirements.txt -r requirements-ml.txt
$env:PYTHONIOENCODING="utf-8"   # Windows; optional encoding fix
python complete_pipeline.py
```

Outputs:

- `models/saved_models/` — saved model weights
- `reports/model_evaluation_results.csv` — evaluation metrics

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/analyze/image` | Upload image (`multipart/form-data`, field `file`) |
| POST | `/api/analyze/video` | Upload video |
| POST | `/api/analyze/audio` | Upload audio |
| GET | `/api/scans` | Recent scan history |
| GET | `/api/scans/{scan_id}` | Single scan detail |
| GET | `/api/analytics` | Dashboard counts |
| GET | `/api/performance` | Model performance summary from CSV reports |

---

## Project Layout

```text
backend/
  server.py          # FastAPI app and local REAL/FAKE logic
  run_local.py       # Starts API and writes frontend/.env.local
frontend/
  src/AppNew.jsx     # Main UI
data/raw/            # Reference samples (you provide on each machine)
models/saved_models/ # Optional trained weights
reports/             # Evaluation CSVs (optional)
```

---

## Build Frontend For Production

```powershell
cd frontend
npm run build
```

Static files are output to `frontend/dist/`. Serve them with any static host; set `VITE_BACKEND_URL` to your deployed API URL at build time if needed.

---

## Important Notes

- **Start the backend before the frontend** so `frontend/.env.local` has the correct API URL.
- If the backend port changes, **restart the frontend**.
- The dev server proxies `/api/*` to the backend when configured via Vite.
- **Scan history is in-memory** and clears when the backend restarts.
- Do not commit `.env`, local uploads, or secrets.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| Frontend cannot reach API | Check `frontend/.env.local`, confirm backend is running, restart frontend |
| `pip install` fails | Use Python 3.10+, activate `.venv`, upgrade pip |
| `npm install` fails | Delete `frontend/node_modules`, run `npm install` again |
| Port 8000 in use | `run_local.py` auto-picks the next port; restart frontend after backend starts |
| “Reference data directory not found” | Create `data/raw/images`, `videos`, `audios` with `real_*` and `fake_*` files |
| Video always INCONCLUSIVE | Add more reference videos; use shorter clips with clear faces; restart backend after updating `server.py` |
| No frames extracted | Try another codec/format (MP4 H.264), longer clip, or less corrupted file |
| Audio analysis fails | Prefer WAV; install codecs or ensure `librosa`/`soundfile` can read the file |

---

## Sharing The Project With Others

To hand off to a teammate or run on a second machine:

1. Push your code to GitHub (do **not** push `.env` or secrets). The `data/` reference set is tracked in this repository.
2. Send them this README’s **“Set Up On Another Local Machine”** section.
3. They clone → venv → `pip install -r backend/requirements.txt` → run backend + frontend (dataset comes from the clone).

---

## License & Repository

- Git remote: "https://github.com/Vaishnavi-Chougale/RealityShield-AI-Generated-Content-Detector.git"
- Developed by: Vaishnavi Namadev Chougale