# Hand Landmarker model

The demo uses MediaPipe's **Hand Landmarker** task bundle:

- **Official documentation:** <https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker>
- **Download URL:** <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>
- **File:** `hand_landmarker.task` (float16, 1 hand, 21 landmarks)
- **SHA-256:** `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- **License:** Apache License 2.0, as distributed by Google with the MediaPipe model bundle.

The model file is intentionally **not committed**; download it with:

```bash
.venv/bin/python scripts/download_model.py
```

The script verifies the checksum before replacing the file. If you prefer to
download it manually, place it at `models/hand_landmarker.task` and verify the
checksum:

```bash
sha256sum models/hand_landmarker.task
```
