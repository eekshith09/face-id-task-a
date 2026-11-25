"""API tests with a fake engine, so they run fast and without the model files."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.face_engine import UNKNOWN, Gallery, GalleryEntry
from app.main import create_app


def png_bytes():
    return cv2.imencode(".png", np.full((50, 80, 3), 255, dtype=np.uint8))[1].tobytes()


class FakeEngine:
    """Pretends every readable reference image has a face, and the group photo has two faces."""

    def __init__(self, faces_in_group=2):
        self.faces_in_group = faces_in_group

    def load(self):
        pass

    def build_gallery(self, files):
        g = Gallery()
        for name, data in files:
            if data == b"no-face":
                g.warnings.append(f"{name}: no face found, skipped.")
            else:
                g.entries.append(GalleryEntry(name.split(".")[0].title(), name, np.ones(512)))
        return g

    def identify(self, image, gallery, threshold):
        faces = [{"id": 1, "name": gallery.people[0], "score": 0.8, "box": [1, 2, 10, 12]},
                 {"id": 2, "name": UNKNOWN, "score": 0.1, "box": [20, 2, 30, 12]}]
        return faces[: self.faces_in_group]


@pytest.fixture
def client():
    return TestClient(create_app(engine=FakeEngine(), preload_model=False))


def upload_gallery(client, files):
    return client.post("/api/gallery", files=[("files", (n, d, "image/png")) for n, d in files])


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Who Is in the Photo" in r.text


def test_identify_before_gallery_is_rejected(client):
    r = client.post("/api/identify", files={"file": ("g.png", png_bytes(), "image/png")})
    assert r.status_code == 400 and "reference folder" in r.json()["detail"]


def test_gallery_with_no_images_is_rejected(client):
    r = upload_gallery(client, [("notes.txt", b"hello"), (".DS_Store", b"x")])
    assert r.status_code == 400


def test_gallery_with_no_faces_is_rejected_and_keeps_old_gallery(client):
    assert upload_gallery(client, [("alice.png", b"img")]).status_code == 200
    r = upload_gallery(client, [("bob.png", b"no-face")])
    assert r.status_code == 400
    assert client.get("/api/status").json()["people"] == ["Alice"]


def test_gallery_reports_ignored_files_and_warnings(client):
    r = upload_gallery(client, [("alice.png", b"img"), ("bob.png", b"no-face"), ("readme.txt", b"x")])
    body = r.json()
    assert r.status_code == 200
    assert body["people"] == ["Alice"] and body["ignored_files"] == 1
    assert any("bob.png" in w for w in body["warnings"])


def test_identify_invalid_image(client):
    upload_gallery(client, [("alice.png", b"img")])
    r = client.post("/api/identify", files={"file": ("g.jpg", b"garbage", "image/jpeg")})
    assert r.status_code == 400 and "not a readable image" in r.json()["detail"]


def test_identify_success(client):
    upload_gallery(client, [("alice.png", b"img")])
    r = client.post("/api/identify", files={"file": ("g.png", png_bytes(), "image/png")})
    body = r.json()
    assert r.status_code == 200
    assert (body["width"], body["height"]) == (80, 50)
    assert body["summary"] == {"total": 2, "identified": 1, "unknown": 1}


def test_identify_no_faces():
    client = TestClient(create_app(engine=FakeEngine(faces_in_group=0), preload_model=False))
    upload_gallery(client, [("alice.png", b"img")])
    r = client.post("/api/identify", files={"file": ("g.png", png_bytes(), "image/png")})
    assert r.status_code == 200 and r.json()["summary"]["total"] == 0
