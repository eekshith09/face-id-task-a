"""Face detection, gallery building and matching.

This module has no web code so it can be used and tested on its own.

Pipeline:
  1. Reference photos -> one embedding per photo, labelled with a name from the filename.
  2. Group photo -> detect every face -> one embedding per face.
  3. Compare with cosine similarity; accept the best match only if it clears a threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Starting value for buffalo_l (ArcFace) cosine similarity. NOT tuned on evaluation data;
# see docs/DEVELOPMENT_LOG.md. Higher = stricter (fewer wrong names, more "Unknown").
DEFAULT_THRESHOLD = 0.40

UNKNOWN = "Unknown"


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


# ---------------------------------------------------------------------------
# Small pure helpers (no model needed)
# ---------------------------------------------------------------------------

def name_from_filename(filename: str) -> str:
    """Turn a reference filename into a display name.

    'john_smith.jpg' -> 'John Smith'; 'Priya-Patel_2.png' -> 'Priya Patel';
    'Anna (3).jpeg' -> 'Anna'. Trailing numbers let one person have several photos.
    """
    stem = Path(filename).stem
    stem = re.sub(r"[\s_\-.]*\(?\d+\)?$", "", stem)  # drop trailing "_2", " (3)", etc.
    words = [w for w in re.split(r"[\s_\-.]+", stem) if w]
    # Capitalise all-lowercase words only, so names like "McDonald" are kept as typed.
    words = [w.capitalize() if w.islower() else w for w in words]
    return " ".join(words)


def is_supported_image(filename: str) -> bool:
    name = Path(filename).name
    return not name.startswith(".") and Path(name).suffix.lower() in SUPPORTED_EXTENSIONS


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR image, or raise InvalidImageError."""
    if not data:
        raise InvalidImageError("The file is empty.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # also applies EXIF rotation
    if image is None:
        raise InvalidImageError("The file is not a readable image.")
    return image


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def match_faces(
    face_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_names: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[tuple[str, float]]:
    """Match each face to a person name, or UNKNOWN.

    face_embeddings: (n_faces, d); gallery_embeddings: (n_refs, d); gallery_names: n_refs names
    (a person may appear several times if they have several reference photos).

    Returns one (name, score) per face. `score` is the cosine similarity to the assigned
    person, or for unknown faces the best similarity seen (useful for tuning the threshold).

    A person can be assigned to at most one face per photo: the highest-scoring pairs are
    accepted first, so the same name never appears on two different faces.
    """
    n_faces = len(face_embeddings)
    if n_faces == 0:
        return []
    if len(gallery_embeddings) == 0:
        return [(UNKNOWN, 0.0)] * n_faces

    sims = normalize(face_embeddings) @ normalize(gallery_embeddings).T  # (n_faces, n_refs)

    # Collapse several reference photos per person into one score: the best one.
    people = sorted(set(gallery_names))
    names_arr = np.array(gallery_names)
    person_sims = np.stack(
        [sims[:, names_arr == person].max(axis=1) for person in people], axis=1
    )  # (n_faces, n_people)

    results: list[tuple[str, float]] = [
        (UNKNOWN, float(person_sims[i].max())) for i in range(n_faces)
    ]

    # Greedy one-to-one assignment, strongest pairs first.
    candidates = [
        (person_sims[i, j], i, j)
        for i in range(n_faces)
        for j in range(len(people))
        if person_sims[i, j] >= threshold
    ]
    candidates.sort(reverse=True)
    used_faces: set[int] = set()
    used_people: set[int] = set()
    for score, i, j in candidates:
        if i in used_faces or j in used_people:
            continue
        results[i] = (people[j], float(score))
        used_faces.add(i)
        used_people.add(j)
    return results


# ---------------------------------------------------------------------------
# Model-backed engine
# ---------------------------------------------------------------------------

@dataclass
class GalleryEntry:
    name: str
    filename: str
    embedding: np.ndarray


@dataclass
class Gallery:
    entries: list[GalleryEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def people(self) -> list[str]:
        return sorted({e.name for e in self.entries})

    def embeddings(self) -> np.ndarray:
        if not self.entries:
            return np.zeros((0, 512), dtype=np.float32)
        return np.stack([e.embedding for e in self.entries])

    def names(self) -> list[str]:
        return [e.name for e in self.entries]


class FaceEngine:
    """Wraps InsightFace's buffalo_l model (RetinaFace detector + ArcFace recogniser)."""

    def __init__(self, model_root: str | Path, det_size: tuple[int, int] = (640, 640)):
        self.model_root = str(model_root)
        self.det_size = det_size
        self._app = None

    def load(self):
        # Loaded lazily so importing this module (e.g. in tests) stays fast.
        if self._app is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name="buffalo_l",
                root=self.model_root,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=self.det_size)
            self._app = app
        return self._app

    def detect(self, image: np.ndarray) -> list[dict]:
        """Return detected faces as dicts with 'box' [x1, y1, x2, y2], 'det_score', 'embedding'."""
        faces = self.load().get(image)
        return [
            {
                "box": [float(v) for v in f.bbox],
                "det_score": float(f.det_score),
                "embedding": np.asarray(f.normed_embedding, dtype=np.float32),
            }
            for f in faces
        ]

    def build_gallery(self, files: list[tuple[str, bytes]]) -> Gallery:
        """Build a gallery from (filename, bytes) pairs. Problem files are skipped with a warning."""
        gallery = Gallery()
        for filename, data in files:
            if not is_supported_image(filename):
                continue  # silently ignore non-images such as .DS_Store or .txt
            name = name_from_filename(filename)
            if not name:
                gallery.warnings.append(f"{filename}: no name in filename, skipped.")
                continue
            try:
                image = decode_image(data)
            except InvalidImageError as exc:
                gallery.warnings.append(f"{filename}: {exc} Skipped.")
                continue
            faces = self.detect(image)
            if not faces:
                # Tightly cropped headshots often fail detection because the detector
                # expects some background around the face. Retry with a 25% border.
                pad = int(max(image.shape[:2]) * 0.25)
                padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                faces = self.detect(padded)
            if not faces:
                gallery.warnings.append(f"{filename}: no face found, skipped.")
                continue
            if len(faces) > 1:
                gallery.warnings.append(
                    f"{filename}: {len(faces)} faces found, using the largest one."
                )
            largest = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
            gallery.entries.append(GalleryEntry(name, filename, largest["embedding"]))
        return gallery

    def identify(self, image: np.ndarray, gallery: Gallery, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
        """Detect faces in a group photo and label each one with a name or UNKNOWN."""
        faces = self.detect(image)
        if not faces:
            return []
        matches = match_faces(
            np.stack([f["embedding"] for f in faces]),
            gallery.embeddings(),
            gallery.names(),
            threshold,
        )
        # Sort left-to-right so face numbers follow reading order in the photo.
        order = sorted(range(len(faces)), key=lambda i: faces[i]["box"][0])
        return [
            {
                "id": n + 1,
                "name": matches[i][0],
                "score": round(matches[i][1], 3),
                "box": [round(v, 1) for v in faces[i]["box"]],
            }
            for n, i in enumerate(order)
        ]
