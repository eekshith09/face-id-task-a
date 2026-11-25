"""FastAPI web server: serves the page and two JSON endpoints.

Run with:  .venv/bin/uvicorn app.main:app --port 8765
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.face_engine import (
    DEFAULT_THRESHOLD,
    UNKNOWN,
    FaceEngine,
    Gallery,
    InvalidImageError,
    decode_image,
    is_supported_image,
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODEL_ROOT = Path(os.environ.get("FACEID_MODEL_ROOT", BASE_DIR / "models"))
THRESHOLD = float(os.environ.get("FACEID_THRESHOLD", DEFAULT_THRESHOLD))
MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB per image
MAX_GALLERY_FILES = 500


def create_app(engine=None, preload_model: bool = True) -> FastAPI:
    """Build the app. Tests pass a fake engine so they do not need the real model."""
    engine = engine or FaceEngine(MODEL_ROOT)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if preload_model:
            engine.load()  # load once at startup so the first request is not slow
        yield

    app = FastAPI(title="Who Is in the Photo?", lifespan=lifespan)
    app.state.engine = engine
    app.state.gallery = Gallery()  # in memory; one shared gallery for this local app

    def read_upload(upload: UploadFile) -> bytes:
        data = upload.file.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, f"{upload.filename}: file is larger than 15 MB.")
        return data

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status(request: Request):
        gallery: Gallery = request.app.state.gallery
        return {"people": gallery.people, "reference_photos": len(gallery.entries), "threshold": THRESHOLD}

    @app.post("/api/gallery")
    def load_gallery(request: Request, files: list[UploadFile] = File(...)):
        images = [f for f in files if is_supported_image(f.filename or "")]
        if not images:
            raise HTTPException(400, "No image files (.jpg, .jpeg, .png, .bmp, .webp) were found in the selected folder.")
        if len(images) > MAX_GALLERY_FILES:
            raise HTTPException(400, f"Too many images ({len(images)}). The limit is {MAX_GALLERY_FILES}.")

        gallery = request.app.state.engine.build_gallery([(f.filename, read_upload(f)) for f in images])
        if not gallery.entries:
            raise HTTPException(
                400,
                {"message": "No usable faces were found in the reference photos.", "warnings": gallery.warnings},
            )
        request.app.state.gallery = gallery  # replace only when the new gallery is usable
        return {
            "people": gallery.people,
            "reference_photos": len(gallery.entries),
            "ignored_files": len(files) - len(images),
            "warnings": gallery.warnings,
        }

    @app.post("/api/identify")
    def identify(request: Request, file: UploadFile = File(...)):
        gallery: Gallery = request.app.state.gallery
        if not gallery.entries:
            raise HTTPException(400, "Load a reference folder first.")
        try:
            image = decode_image(read_upload(file))
        except InvalidImageError as exc:
            raise HTTPException(400, f"{file.filename}: {exc}")

        faces = request.app.state.engine.identify(image, gallery, THRESHOLD)
        height, width = image.shape[:2]
        known = sum(1 for f in faces if f["name"] != UNKNOWN)
        return {
            "width": width,
            "height": height,
            "threshold": THRESHOLD,
            "faces": faces,
            "summary": {"total": len(faces), "identified": known, "unknown": len(faces) - known},
        }

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
