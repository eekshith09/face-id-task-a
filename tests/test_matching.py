"""Tests for the pure helpers. These use synthetic vectors, so the model is not needed."""

from collections import Counter

import cv2
import numpy as np
import pytest

from app.face_engine import (
    UNKNOWN,
    InvalidImageError,
    decode_image,
    is_supported_image,
    match_faces,
    name_from_filename,
)


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("john_smith.jpg", "John Smith"),
        ("Priya-Patel_2.png", "Priya Patel"),
        ("Anna (3).jpeg", "Anna"),
        ("mary.jane.watson.JPG", "Mary Jane Watson"),
        ("McDonald.jpg", "McDonald"),
        ("people/sub/li_wei_01.webp", "Li Wei"),
        ("12.jpg", ""),
    ],
)
def test_name_from_filename(filename, expected):
    assert name_from_filename(filename) == expected


def test_is_supported_image():
    assert is_supported_image("a.JPG")
    assert is_supported_image("folder/b.png")
    assert not is_supported_image(".DS_Store")
    assert not is_supported_image("notes.txt")
    assert not is_supported_image("._hidden.jpg")


def test_decode_image_rejects_garbage():
    with pytest.raises(InvalidImageError):
        decode_image(b"this is not an image")
    with pytest.raises(InvalidImageError):
        decode_image(b"")


def test_decode_image_accepts_png():
    ok, buf = cv2.imencode(".png", np.zeros((10, 20, 3), dtype=np.uint8))
    assert ok
    assert decode_image(buf.tobytes()).shape == (10, 20, 3)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_match_known_and_unknown():
    gallery = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    names = ["Alice", "Bob"]
    faces = np.stack([
        _unit([0.9, 0.1, 0]),  # close to Alice
        _unit([0, 0, 1]),      # nobody
    ])
    result = match_faces(faces, gallery, names, threshold=0.5)
    assert result[0][0] == "Alice" and result[0][1] > 0.9
    assert result[1][0] == UNKNOWN


def test_match_uses_best_of_several_reference_photos():
    gallery = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    names = ["Alice", "Alice"]  # two photos of the same person
    faces = np.stack([_unit([0.05, 1, 0])])
    assert match_faces(faces, gallery, names, threshold=0.5)[0][0] == "Alice"


def test_one_person_is_not_assigned_to_two_faces():
    gallery = np.stack([_unit([1, 0, 0])])
    faces = np.stack([_unit([1, 0.1, 0]), _unit([1, 0.3, 0])])  # both resemble Alice
    result = match_faces(faces, gallery, ["Alice"], threshold=0.5)
    assert [r[0] for r in result] == ["Alice", UNKNOWN]  # the closer face wins


def test_threshold_is_inclusive_boundary():
    gallery = np.stack([_unit([1, 0])])
    faces = np.stack([_unit([0.6, 0.8])])  # cosine = 0.6
    assert match_faces(faces, gallery, ["A"], threshold=0.6)[0][0] == "A"
    assert match_faces(faces, gallery, ["A"], threshold=0.61)[0][0] == UNKNOWN


def test_empty_inputs():
    assert match_faces(np.zeros((0, 3)), np.zeros((1, 3)), ["A"]) == []
    assert match_faces(np.ones((2, 3)), np.zeros((0, 3)), []) == [(UNKNOWN, 0.0), (UNKNOWN, 0.0)]


def test_flat_folder_filenames_group_into_one_identity_per_person():
    # Same naming pattern as the real flat reference folder: <name><number>.<ext>,
    # no separator before the number, mixed .jpg / .JPG extensions.
    filenames = [
        f"{prefix}{i}{ext}"
        for prefix, ext in [("akshaya", ".jpg"), ("eekshith", ".JPG"), ("jeevani", ".JPG"),
                            ("navaneeth", ".JPG"), ("nikita", ".JPG"), ("nikith", ".JPG")]
        for i in range(1, 6)
    ]
    groups = Counter(name_from_filename(f) for f in filenames)
    assert groups == {
        "Akshaya": 5, "Eekshith": 5, "Jeevani": 5,
        "Navaneeth": 5, "Nikita": 5, "Nikith": 5,
    }


def test_several_people_with_several_photos_are_matched_per_person():
    # Two people, two reference photos each, deliberately interleaved in the gallery.
    alice_1, alice_2 = _unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])
    bob_1, bob_2 = _unit([0, 0, 1, 0]), _unit([0, 0, 0, 1])
    gallery = np.stack([alice_1, bob_1, alice_2, bob_2])
    names = ["Alice", "Bob", "Alice", "Bob"]

    faces = np.stack([
        _unit([0, 1, 0.2, 0]),       # resembles only Alice's 2nd photo
        _unit([0, 0.2, 0, 1]),       # resembles only Bob's 2nd photo
        _unit([0.5, 0.5, 0.5, 0.5]), # equally (weakly) similar to every photo
    ])
    result = match_faces(faces, gallery, names, threshold=0.6)

    assert [name for name, _ in result] == ["Alice", "Bob", UNKNOWN]
    # Score is the best single photo of that person (max), not the average over photos:
    # an average would be ~0.49 here, below the threshold, and the face would be Unknown.
    assert result[0][1] == pytest.approx(0.981, abs=1e-3)
    assert result[1][1] == pytest.approx(0.981, abs=1e-3)
    assert result[2][1] == pytest.approx(0.5, abs=1e-6)
