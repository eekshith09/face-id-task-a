# Development Log

This records decisions and test results as they happened. Each entry is either a measured
outcome or a decision with its reason. Nothing listed as "pending" has been done yet.

Built from scratch in a new, empty folder on 2026-11-25, working with an AI coding assistant
(Claude Code) in the terminal.

## Environment found at start (2026-11-25)

- Folder contained only a `.venv` and an empty git repo with no commits.
- Python 3.12.16 on macOS arm64. Already installed: fastapi 0.165.0, insightface 0.7.3,
  onnxruntime 1.32.0, opencv-python-headless 5.1.0.95, numpy 2.5.5, pytest 9.2.0.
- `import insightface, onnxruntime, cv2, fastapi` worked. onnxruntime providers available:
  CoreML, Azure, CPU.
- The buffalo_l model was **not** present.

## Stage 1: model download

- Downloaded buffalo_l through InsightFace's own downloader from
  `github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip` (281,857 KB) into
  `./models/models/buffalo_l/`, which git ignores. Download plus first load took 180 s. Unzipped
  size on disk: 614 MB.
- Decision: keep the model **inside the project folder** (`./models`) instead of
  `~/.insightface`, so the project is self-contained.
- Decision: load only the `detection` and `recognition` modules. The landmark and gender/age
  models aren't needed. Use **CPUExecutionProvider** only, so behaviour is the same on any machine.

## Stage 2: face engine (`app/face_engine.py`)

Decisions:
- **Cosine similarity** on L2-normalised ArcFace embeddings. After normalising, it is just a dot
  product, so the whole comparison is one NumPy matrix multiply.
- **Threshold 0.40.** This is a commonly used starting point for buffalo_l. It has **not** been
  tuned on evaluation data. It can be changed with `FACEID_THRESHOLD`.
- **Several photos per person:** the person's score is the max over their photos (not an average).
  This is simple to explain, and a single good photo is enough.
- **One person per face per photo:** greedy assignment, strongest pairs first, so the same name
  can't appear on two faces. If a face loses its best person to a stronger face, it is labelled
  Unknown rather than being given a weaker second choice.
- **Name parsing:** remove the extension and trailing numbers like `_2` or ` (3)`, split on
  `_ - . space`, and capitalise all-lowercase words only (so `McDonald` stays as typed).
- **Reference photo with several faces:** use the largest face and add a warning.
- **Invalid or unreadable images** raise `InvalidImageError` and become user-facing messages,
  never a crash.

Tests: `tests/test_matching.py`, 15 tests (name parsing, file filtering, image decoding, known vs
unknown, best of several photos, one-to-one assignment, threshold boundary, empty inputs).
Result: **15 passed**.

Smoke test with the real model, using InsightFace's bundled sample images:
- `t1.jpg` (1280×886): **6 faces detected**, 2.99 s for the first inference on CPU.
- Gallery made of 2 faces cropped from `t1.jpg`, plus `Tom_Hanks_54745.png`, a broken file and a
  `.txt` file. Identifying `t1.jpg` gave the 2 cropped people at 0.981 and 0.987, and the other 4
  faces Unknown (best scores 0.064, 0.057, 0.217, 0.101). The broken file was skipped with a
  warning and the `.txt` file was ignored.
- This only checks that the pipeline works end to end. It is **not** an accuracy test, because
  the references came from the same photo.

Problem found: `Tom_Hanks_54745.png` gave **no face detected**. It is a 112×112 tight face crop.
Adding a black border of 25% or 50% of the image size made the detector find 1 face in both cases.
- Decision: if no face is found in a **reference** photo, retry once with a 25% border. This is
  useful for tightly cropped ID-style photos. It isn't applied to group photos.

## Stage 3: API (`app/main.py`)

Decisions:
- **Folder selection happens in the browser** (`<input webkitdirectory>`), which uploads the
  folder's files. Staff don't have to type server paths, and only the filename is used for the name.
- Endpoints: `POST /api/gallery`, `POST /api/identify`, `GET /api/status`.
- A gallery that has no usable faces is rejected, and the **previous gallery is kept**.
- Limits: 15 MB per image, 500 images per gallery.
- `create_app(engine=...)` takes the engine as a parameter, so the API tests use a fake engine
  without the model.
- Endpoints are plain `def` (not `async`), so CPU-heavy inference runs in FastAPI's thread pool.

Tests: `tests/test_api.py`, 8 tests (page served, identify before gallery, folder with no
images, gallery with no faces keeps the old one, ignored files and warnings reported, invalid
image, success, no faces). Full suite: **23 passed**.

Problem found: port 8000 was already used by Docker on this machine, so `localhost:8000` returned
Docker's 404s. Decision: use port **8765** and bind to `127.0.0.1`.

Manual HTTP test with curl against the real model on 127.0.0.1:8765:
- Identify before loading a gallery: `400 "Load a reference folder first."`
- Gallery (person_a.jpg, person_b.jpg, tom_hanks.png, notes.txt): 3 people loaded,
  `ignored_files: 1`, no warnings. Tom Hanks now loads because of the border retry.
- Identify `t1.jpg`: 6 faces, 2 identified, 4 unknown. Same scores as the smoke test. The request
  took about 0.3 s once the model was loaded.
- Text file uploaded as `fake.jpg`: `400 "fake.jpg: The file is not a readable image."`
- `/`, `/static/app.js`, `/static/style.css`: all 200.

## Stage 4: frontend (`app/static/`)

- Plain HTML/CSS/JS with two numbered steps and a results card. No external fonts, scripts or
  CDNs, so the page needs no internet.
- Boxes and labels are drawn on a `<canvas>`: green for identified faces, orange for Unknown,
  numbered left to right. The results table shows the same numbers, names and similarity scores.
- Similarity is shown as a raw score (for example 0.62), not as a percentage, so it isn't
  mistaken for an accuracy figure.
- `node --check app/static/app.js`: syntax OK.

## Pending (not yet done)

- [ ] Manual browser test of the full UI (folder picker → group photo → boxes and table).
- [ ] Test with real reference photos and group photos of real people.
- [ ] **Offline test with Wi-Fi turned off.** Not done yet, so offline operation is not claimed.
- [ ] Threshold evaluation on a labelled set before quoting any accuracy.
