# AI Collaboration Record

## Purpose and evidence standard

The assignment asks for 5–10 key decisions, evidence of how the candidate steered the AI, 2–3 pasted chat excerpts, and the full AI chat history from the first prompt.

This file records decisions that are supported by the project development notes and conversation context available at the time this document was written. It does **not** invent verbatim AI replies or claim to contain a complete export.

**Outstanding evidence:** the complete ChatGPT and Claude Code conversation exports, beginning with the first prompt, have not yet been added to this repository. The 2–3 verbatim paired chat excerpts requested by the brief also need to be pasted from the original conversation exports. Add the actual exported files under `docs/chat-history/` (or add valid share links) before describing the AI-history requirement as complete.

## Key decisions

### Decision 1 — Build a visual browser workflow

- **AI-assisted direction:** A local web application with a folder picker and group-photo upload was explored as a practical interface.
- **Candidate decision:** Use a FastAPI backend with a plain HTML/CSS/JavaScript frontend.
- **Why:** The task is intended for office users; they should select a folder, upload a photo and inspect labels without using a command-line workflow for each identification.
- **Evidence:** Current implementation has a browser directory picker, upload flow, canvas annotations and a results list.

### Decision 2 — Keep inference local

- **AI-assisted direction:** Use a locally installed face-recognition model rather than a hosted inference API.
- **Candidate decision:** Use InsightFace `buffalo_l` and run the app on the laptop.
- **Why:** The assignment explicitly requires fully offline runtime and prohibits online APIs or hosted services inside the solution.
- **Evidence:** `app/face_engine.py` loads InsightFace locally; the README documents a one-time model download. A Wi-Fi-disabled end-to-end test is still pending, so offline compliance is not yet verified.

### Decision 3 — Use filenames to identify reference people

- **AI-assisted direction:** Derive the identity label from each reference image filename and allow multiple photos for one identity.
- **Candidate decision:** Use a flat folder with numbered filenames for multiple photos, and group files by the parsed name.
- **Why:** This matches the task's named-photo-folder input and allows several reference views of a person.
- **Evidence:** `name_from_filename()` and the matching tests cover names, suffix numbering and multiple reference photos.

### Decision 4 — Use cosine similarity over ArcFace embeddings

- **AI-assisted direction:** Compare face embeddings using a similarity score and reject weak matches.
- **Candidate decision:** Use normalized ArcFace embeddings and cosine similarity with a configurable default threshold of 0.40.
- **Why:** The score is straightforward to compute and explain. The threshold is configurable so it can be tuned after collecting labelled evaluation data.
- **Evidence:** `match_faces()` and tests cover known/unknown decisions and the inclusive threshold boundary. The 0.40 value has not been tuned and is not an accuracy guarantee.

### Decision 5 — Support multiple reference photos per person

- **AI-assisted direction:** Aggregate or compare multiple reference photos for one person.
- **Candidate decision:** Score a person using the maximum similarity across that person's reference photos.
- **Why:** One reference photo may represent a pose or lighting condition better than another.
- **Evidence:** `test_match_uses_best_of_several_reference_photos` and `test_several_people_with_several_photos_are_matched_per_person`.

### Decision 6 — Prevent duplicate identity assignments in one group photo

- **AI-assisted direction:** Apply one-to-one assignment between detected faces and gallery identities.
- **Candidate decision:** Use greedy assignment in descending similarity order.
- **Why:** A single person should not be labelled on two different faces in the same group photo, and the greedy method is simple to inspect.
- **Evidence:** `test_one_person_is_not_assigned_to_two_faces`. The trade-off is that greedy assignment is not guaranteed to be globally optimal.

### Decision 7 — Preserve unknown people instead of forcing a name

- **AI-assisted direction:** Reject a candidate when its similarity is below the threshold.
- **Candidate decision:** Label unmatched faces as `Unknown`.
- **Why:** The task explicitly includes people who are not in the reference folder. A forced incorrect identity is worse than an unknown result.
- **Evidence:** Matching tests assert that unrelated synthetic embeddings return `Unknown`; the sample-photo smoke test also produced unknown detections. Real-world unknown-person performance is not yet evaluated.

### Decision 8 — Keep personal images and model artifacts out of Git

- **AI-assisted direction:** Use ignore rules for local datasets, downloaded model weights, environments and secrets.
- **Candidate decision:** Keep the 89-photo development gallery and model files local; publish source and documentation only.
- **Why:** Personal face photos and biometric artifacts should not be placed in a public repository, and large model files do not belong in the source history.
- **Evidence:** The four initial commits were checked and the public repository was reviewed for photos, model files, embeddings, environment files and secrets. The test file does contain first names in a filename-parser test; this is disclosed in the repository history discussion and can be replaced with placeholders in a later commit if desired.

## Chat excerpts

The assignment asks for 2–3 verbatim excerpts showing the actual interaction. They are intentionally **not fabricated here**. Paste the exact excerpts from the original ChatGPT and Claude Code conversations below, retaining both the candidate's message and the AI response.

### Excerpt 1 — initial problem framing

> Pending: paste the exact original prompt and AI response from the conversation export.

### Excerpt 2 — technical decision

> Pending: paste the exact original prompt and AI response that led to the local InsightFace / FastAPI implementation.

### Excerpt 3 — testing and Git decisions

> Pending: paste the exact original prompt and AI response showing the testing and one-commit-at-a-time review process.

## Full AI conversation history

**Not yet attached.** Add the unedited exports from the first prompt onward. Suggested repository structure:

```text
docs/
  chat-history/
    chatgpt/
      conversation-1.html (or .json / .md export)
    claude-code/
      session-1.txt (or exported transcript)
```

Do not replace the full transcripts with this summary. Once the real exports are added, update this file with links to them and replace the pending excerpt placeholders with exact excerpts.

## Reflection

AI tools were used as development collaborators, but implementation choices, constraints, approvals and test decisions were reviewed by the candidate. The project history is preserved as separate commits rather than rewritten into a single commit. The remaining evidence gaps are explicitly listed above so the submission does not overstate the completeness of the AI collaboration record.
