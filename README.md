# Who Is in the Photo?

A small local web app for BLP Industry.AI Task A. Choose a folder of reference photos, upload a
group photo, and the app boxes each face and labels it with a name or **Unknown**.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): components, data flow, trade-offs, and the Archify
  architecture diagram ([SVG](docs/architecture/face-id-architecture.svg),
  [interactive HTML](docs/architecture/face-id-architecture.html))
- [RESEARCH.md](RESEARCH.md): technologies, papers, alternatives, licensing, development data
- [AI_COLLAB.md](AI_COLLAB.md): AI collaboration record. Full chat exports are still pending.
- [EVALUATION.md](EVALUATION.md): test results and evaluation status. Browser, Wi-Fi-off and
  recognition-accuracy evaluation are still pending.
- [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md): decisions and test results in order

## How it works

1. **Reference folder.** Each image's filename gives the person's name:
   `john_smith.jpg` → "John Smith". Trailing numbers allow several photos per person:
   `john_smith_2.jpg`. Each photo becomes one 512-number face embedding. If a photo has several
   faces, the largest one is used. Photos with no face, or that can't be read, are skipped and
   listed as warnings.
2. **Group photo.** InsightFace `buffalo_l` detects every face (SCRFD detector) and
   computes an embedding for each one (ArcFace recogniser).
3. **Matching.** Faces are compared to reference embeddings with cosine similarity. A person's
   score is their best-matching reference photo. A name is given only when the score is at least
   the threshold (default **0.40**). Each person can be assigned to only one face per photo,
   strongest pairs first. Every other face is **Unknown**.

```
app/face_engine.py   detection, gallery, matching (no web code)
app/main.py          FastAPI: /api/gallery, /api/identify, /api/status, serves the page
app/static/          HTML/CSS/JS page (no external CDN or remote resources)
tests/               pytest: matching logic and API error handling
scripts/download_model.py   one-time model download
```

## Setup (one time, needs internet)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_model.py   # ~280 MB download into ./models
```

## Run

```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765 in a browser that supports folder selection (`webkitdirectory`),
such as Chrome or Edge. Browser testing is still pending.

Optional settings (environment variables):
- `FACEID_THRESHOLD`: matching threshold (default 0.40). Higher means fewer wrong names but more Unknowns.
- `FACEID_MODEL_ROOT`: where the model lives (default `./models`).

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests use synthetic embeddings and a fake engine, so they don't need the model.

## Limitations

- The 0.40 threshold has **not been tuned** on an evaluation set, and no accuracy figure is claimed.
- The gallery is held in memory and is lost when the server restarts. There is one shared
  gallery, so the app is meant for local, single-user use.
- It runs on CPU only.
- The InsightFace pretrained models are licensed for **non-commercial research use**. Commercial
  use would need a licence or a different model.
