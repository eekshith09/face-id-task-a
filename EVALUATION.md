# Evaluation and Test Report

## Summary

This report separates automated software tests from face-recognition evaluation. Passing unit/API tests does not demonstrate recognition accuracy.

## Automated tests

The latest reported clean-checkout run completed:

- **25 tests passed**
- 17 matching / image-helper tests
- 8 API tests

The tests use synthetic embeddings and a fake engine for API tests. They validate program logic and API behaviour without requiring the pretrained model.

Coverage described in the test suite and development log includes:

- filename parsing and supported image filtering;
- invalid and valid image decoding;
- known and unknown matching cases;
- best-of-several-reference-photo matching;
- one-to-one identity assignment;
- threshold boundary behaviour;
- empty inputs;
- API page serving, identification before gallery loading, invalid uploads, empty galleries, warnings, and successful response handling.

Re-run from the repository root:

```bash
.venv/bin/python -m pytest -q
```

The 25-pass result is the previously recorded result, not a claim that the suite was rerun while creating this document.

## Real-model smoke test

A smoke test was recorded using InsightFace's bundled sample group image `t1.jpg`:

- Image dimensions: 1280 × 886.
- Faces detected: 6.
- Two reference crops taken from that same image matched at cosine scores approximately 0.981 and 0.987.
- The remaining four detected faces were labelled `Unknown`, with best scores recorded as approximately 0.064, 0.057, 0.217 and 0.101.
- First CPU inference took approximately 2.99 seconds.
- After the model was loaded, a later API identify request took approximately 0.3 seconds.

**Interpretation:** this confirms that model loading, detection, embedding extraction and matching execute end to end on the sample. Because the reference crops came from the same source group image, it is not an independent recognition benchmark and must not be presented as accuracy.

## Reference gallery load

The prepared local development reference folder was reported to contain 89 supported single-face images across six identities:

| Identity | Images |
|---|---:|
| Akshaya | 7 |
| Eekshith | 32 |
| Jeevani | 7 |
| Navaneeth | 10 |
| Nikita | 19 |
| Nikith | 14 |
| **Total** | **89** |

The reported load time was approximately 20.0 seconds. The files were copied byte-for-byte and checksummed; the source images were not modified. These are personal development photos and are not included in the public repository.

This gallery-load check is not a recognition accuracy evaluation. The application has not yet been independently evaluated on a labelled set of group photos.

## Evaluation not yet completed

The following checks remain pending and must not be described as passed:

- [ ] Manual browser test: select a reference folder, upload a group photo, inspect boxes and results list.
- [ ] Full end-to-end run with Wi-Fi disabled.
- [ ] Test on an independently labelled group photo containing both known people and people absent from the gallery.
- [ ] Leave-one-out or other held-out evaluation with identity labels and reproducible methodology.
- [ ] Threshold calibration using a separate validation set.
- [ ] Evaluation with the assignment's unseen 50-person folder and approximately 10-person group photo.
- [ ] Record false matches, false unknowns, missed detections and per-image results.

## Metrics

No accuracy, precision, recall, F1 score, false-accept rate or false-reject rate is reported because a suitable independent labelled evaluation set has not yet been tested. The default cosine threshold of 0.40 is an initial configurable value, not a measured optimum.

## Known limitations affecting evaluation

- Reference photos containing multiple faces currently generate a warning and use the largest detected face; they are not rejected.
- Greedy one-to-one matching is not guaranteed to be globally optimal.
- Detection and recognition can be affected by small faces, profile views, occlusion, lighting, image quality and similar-looking people.
- The gallery is held in memory and must be loaded again after a server restart.
- The app's offline behaviour has not yet been confirmed with Wi-Fi disabled.
