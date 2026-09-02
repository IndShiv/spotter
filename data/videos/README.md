# data/videos/

Real recorded clips for the eval harness go here, as a video file plus a
JSON sidecar with the same stem, e.g.:

```
squat_front_01.mp4
squat_front_01.json
```

Sidecar format:

```json
{
  "exercise": "squat",
  "true_count": 8,
  "rep_timestamps": [1.9, 3.4, 4.8, 6.3, 7.7, 9.1, 10.6, 12.0],
  "camera_angle": "front, ~1.5m, eye height",
  "notes": "bodyweight, moderate tempo"
}
```

`rep_timestamps` should mark the moment each rep is judged complete (return
to standing), in seconds from the start of the clip — the same convention
`spotter/eval/synthetic.py` uses for its ground truth, so real and synthetic
clips score identically in `spotter/eval/run.py`.

Video files are gitignored (`data/videos/*.mp4` etc. — see `.gitignore`);
only the JSON sidecars are versioned. `spotter eval` / `make eval` skips any
sidecar whose video isn't present locally rather than failing, so CI runs
fine without them, using the synthetic clip set in `spotter/eval/synthetic.py`
instead.
