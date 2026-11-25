# Research — Who Is in the Photo?

## Research question

How can a laptop application identify people in a group photo using a folder of named reference photos, while keeping all inference local and supporting people who are not in the reference folder as `Unknown`?

The assignment requires a browser-based folder selection workflow, a group-photo upload, face labels on the photo and a results list. Runtime use must be fully offline, with no hosted inference APIs or cloud services.

## Options considered

| Option | What it provides | Decision |
|---|---|---|
| InsightFace with the `buffalo_l` model pack | Local face detection and face-recognition embeddings; Python integration | **Selected.** It is the model currently used by `app/face_engine.py`. |
| Cloud face-recognition APIs | Managed inference without packaging local models | Rejected because the assignment explicitly prohibits online APIs and requires offline runtime. |
| Build a face detector and recognizer from scratch | Full control over the pipeline | Rejected for this take-home scope; it would require training data, model training and significantly more validation. |
| Browser-only JavaScript inference | Could avoid a Python server | Not selected. The current implementation uses Python, OpenCV, InsightFace and FastAPI. |
| FastAPI + plain HTML/CSS/JavaScript | A small local HTTP API and a browser UI without a frontend build system | **Selected.** This is the current repository architecture. |
| Greedy one-to-one matching | Sort candidate face/person pairs by similarity and assign each person at most once | **Selected.** Simple and explainable; not globally optimal. |
| Hungarian/bipartite assignment | Globally optimal one-to-one assignment for a defined score matrix | Not implemented. It remains a possible improvement, not a current feature. |

## Technical references

The following are relevant primary project/documentation sources for the technologies used. They explain the selected components; they do not by themselves establish the accuracy of this application.

1. **InsightFace project** — https://github.com/deepinsight/insightface  
   Source for the InsightFace Python package and model ecosystem used by this application.

2. **InsightFace model zoo** — https://github.com/deepinsight/insightface/tree/master/model_zoo  
   Reference for the available model packs, including `buffalo_l`.

3. **SCRFD: Sample and Computation Redistribution for Efficient Face Detection** — https://arxiv.org/abs/2105.04714  
   Research paper describing SCRFD, the face detector used by the selected model pack.

4. **ArcFace: Additive Angular Margin Loss for Deep Face Recognition** — https://arxiv.org/abs/1801.07698  
   Research paper describing ArcFace, the face-recognition approach associated with the embeddings used by the application.

5. **FastAPI documentation** — https://fastapi.tiangolo.com/  
   Reference for the Python API framework used to expose gallery and identification endpoints.

6. **OpenCV documentation** — https://docs.opencv.org/  
   Reference for image decoding and image operations used in the Python pipeline.

7. **ONNX Runtime documentation** — https://onnxruntime.ai/docs/  
   Reference for the local model execution runtime used by the InsightFace installation.

8. **Archify** — https://github.com/tt-a1i/archify  
   Architecture diagram tool explicitly requested by the assignment. The Archify-generated diagram is in `docs/architecture/`; see `ARCHITECTURE.md`.

## Data used during development

- The local development reference folder was assembled from user-provided personal photos. The latest prepared folder contained 89 supported, single-face images across six identities. It is local development data and is not committed to this public repository.
- The group-photo pipeline smoke test used InsightFace's bundled sample image `t1.jpg`; two reference crops were taken from that same image. This validates that the pipeline runs, but is not an independent test set.
- Automated matching tests use synthetic vectors. They test matching logic and edge cases, not recognition accuracy on real faces.
- No held-out benchmark or independently labelled group-photo evaluation has been completed. No accuracy, precision, recall or F1 score is claimed.

## External implementation inspected

A public GitHub project was inspected as a reference for the general offline face-recognition problem:

- https://github.com/Akshayagunda1105/offline-face-recognition

That repository uses a different stack (YuNet/SFace with a React/Vite frontend). It was treated as an external reference only; its code and documentation were not copied into this project.

## Model licensing

InsightFace's code and its pretrained model weights have different licensing terms. The project repository describes the code as MIT licensed, while the pretrained model packs are subject to non-commercial research restrictions. The current application uses the pretrained `buffalo_l` pack. Review the current upstream licence and model terms before any commercial use:

- https://github.com/deepinsight/insightface
- https://github.com/deepinsight/insightface/tree/master/model_zoo

This project is a take-home demonstration, not a claim that the pretrained weights are cleared for commercial deployment.

## Current research limitations

- The 0.40 cosine threshold is an initial value and has not been calibrated on a labelled validation set.
- The current matcher uses greedy assignment, not Hungarian matching.
- The current gallery builder selects the largest face from a multi-face reference image and emits a warning; it does not yet enforce exactly one face per reference image.
- The application has not yet been manually tested end to end in a browser or with Wi-Fi disabled.
- The live demonstration must be tested using an unseen folder of approximately 50 people and a group photo containing both known and unknown people, as described in the assignment.
