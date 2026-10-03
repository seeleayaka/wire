# Fixed generic port tiling experiment

Frozen before tile inference. No test01 reads, no tile/threshold sweep.

- Keep the existing 3-epoch Dell-only supervised port checkpoint and library
  versions. Previously validation-selected weights and train-seen calibration
  remain retrospective evidence, not blind/cross-device accuracy.
- Tile original images into 1280px squares, stride960 (25% nominal overlap),
  append an end-anchored tile on each axis. No GT crops or device coordinates.
  For 3648x2736 this is 4x3=12 tiles/image, 720 tile passes over 60 images.
- CPU4 threads, inference chunks of 2, imgsz960, conf floor .001, native NMS .7,
  max_det300. Drop boxes within16px of an artificial cut edge, never a true
  image boundary. Class-aware cross-tile NMS .5 retains the highest-confidence
  geometry, not unions; retain suppressed tile support IDs for diagnostics.
- Calibrate only the same 30 train01 normals: threshold=max(.25, p95 of per-image
  maximum confidence after edge rejection and cross-tile NMS). Freeze before val.
- Evaluate all 30 fixed val01 images. Reuse their fingerprint-verified actual
  SIFT H and scoring rectangles from the previous full-frame port report. Rebuild
  valid source warp mask with the same constant-border/erosion rule. Do not rerun
  SIFT or score against untransformed source annotations.
- Existing parents and full-frame accepted port hints remain unchanged. Apply
  the earlier within-parent geometry/validity rules to tiled predictions; remove
  duplicates (IoU>=.5) with existing hints, then retain at most one new highest-
  confidence hint per image. No label-based selection or route/type decisions.
- Main comparison: previous 5 precise fragments, 164 overlap, 71 fault regions,
  37 normal regions vs additive tiled output; report added normal hints/images,
  IoU .1/.5, class-matched one-to-one hint metrics and elapsed/RSS sampling.
- Acceptance: more precise fragments without additional normal hints. Regardless
  of outcome, retain as an isolated experiment, not a default GUI detector.
  An abstention or a negative result is not grounds to lower thresholds on val.
