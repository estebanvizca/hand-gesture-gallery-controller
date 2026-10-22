# Live gesture benchmark

This folder documents the measurement protocol for the gallery. The benchmark
is a guided Pygame session; it is not a unit test, and it cannot be completed
without a person physically performing the gestures.

## Run

From the project root, with the model downloaded and one hand visible:

```bash
python benchmark.py --attempts 20 --idle-minutes 5 \
  --lighting normal --distance-cm 60 \
  --notes "desk lamp, plain wall, seated"
```

Press `Space` on the start screen. The session runs four gesture blocks in
this order: swipe left, swipe right, pinch, fist. Each prompt waits up to
10 seconds for an event (configurable with `--timeout`). Press `Esc` to abort;
partial results are written.

To test one gesture only:

```bash
python benchmark.py --gesture pinch --attempts 20 --idle-minutes 0
```

## What is measured

- **Correct action:** the first confirmed event after the prompt matches the
  requested gesture. A wrong event counts as incorrect and ends that attempt.
- **Latency:** milliseconds from prompt display to the confirmed event.
- **FPS:** the display clock's measured rate, sampled in the log row.
- **False activations:** confirmed events during the idle phase. The idle
  phase instructs the person not to gesture; any event is counted.
- **False activations per minute:** idle events divided by the idle duration.

`reports/benchmark_<timestamp>.summary.json` contains per-gesture accuracy,
median latency, false activations per minute and median FPS. The companion
CSV has one row per attempt and one row per idle false activation.

## Success target

- At least **17 of 20** correct actions per gesture.
- At most **2 false activations in 5 minutes** without intent.

The target requires a real session. The repository does not include fabricated
results. Record the camera, distance, lighting and any relevant notes so the
numbers can be interpreted.

## Log schema

`session_template.csv` defines the columns:

| Column | Meaning |
|---|---|
| `attempt_id` | Sequential attempt number (0 for an idle false activation) |
| `timestamp` | Local ISO-8601 timestamp |
| `gesture` | Requested gesture, or `idle` |
| `expected_event` | Internal label for the requested gesture, or `NONE` |
| `detected_event` | First confirmed event, or `NONE` on timeout |
| `correct` | `1` / `0` |
| `latency_ms` | Prompt-to-event latency; blank on timeout |
| `fps` | Measured FPS at the result row |
| `false_activation` | `1` for idle events, `0` for prompted attempts |
| `lighting` | Free-text lighting description |
| `distance_cm` | Approximate camera distance in cm |
| `camera` | Camera index used |
| `notes` | Session notes |

## Privacy

The benchmark writes **counts only**. It never saves camera frames, video or
images, and it does not transmit anything. The CSV may contain the free-text
notes and lighting description you enter; do not put personal information in
those fields.

## Interpreting a session

- Report each gesture separately. An aggregate across gestures hides a weak
  gesture.
- Report the number of attempts and the exact command-line settings
  (sensitivity, cooldown, camera, model checksum).
- A short session is a pilot, not evidence of general accuracy. Lighting,
  distance, skin tone, hand size and camera quality all matter.
- If the session is aborted, say so and report the number of completed
  attempts; do not extrapolate the missing ones.
