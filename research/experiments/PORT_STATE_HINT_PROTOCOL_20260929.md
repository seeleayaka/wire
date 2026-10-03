# Frozen existing supervised port-state hint experiment

Goal: test a different local evidence source, not another wire-mask/heat threshold
sweep. Keep original coarse CNN-qualified parent geometry and ordering unchanged.

- Existing checkpoint only: output/mendeley_port_state_cpu_poc_20260825/
  yolov8s_rectports_3e_960/weights/best.pt. Originally trained on train01 source
  classes 3/4 as unplugged_plug/unplugged_jack, rectangular mask supervision,
  3 epochs and validation-based checkpoint selection. No new training/download.
- Full image, CPU 4 threads, imgsz960, confidence floor .001 for diagnostic score
  caching, NMS IoU .7, max_det300; no source rectangles or ROI recipes at inference.
- Use the existing 30 train01 normal calibration filenames from the coarse CNN
  protocol; threshold=max(.25, p95 of maximum per-image port detection confidence).
  These normals participated in old supervised training: this is a train-derived
  operating threshold, NOT an independent calibration/generalization guarantee.
- Validation: all fixed 30 val01 images, 15 faults and 15 normals, no test01
  inference, no parameter sweep, no changing threshold after validation results.
- Map predictions and scoring rectangles through the actual production SIFT H
  with seed0, use the source-valid warp coverage. Verify rectangles exactly match
  the frozen cache and source fingerprints match coarse evidence snapshots.
- Each proposed hint must exceed the frozen threshold, be entirely contained in
  a coarse parent, occupy <=half its area, and have >=.98 valid warp coverage.
  Assign to the smallest containing parent, choose highest confidence (stable
  area/coordinate/class ties), at most one hint per parent. No label-based routing.
- Preserve all parents; score parents and parents+hint boxes, image recall,
  fragment overlap, IoU>=.1/.5, class-matched hint precision/recall at .5, added
  normal hints, affected normal images and total review burden. Report moderate
  overlap separately from precise localization; hints are not auto fault verdicts.
- Acceptance for optional integration requires additional precise hits without
  new normal hints on the fixed validation set. Even if passed, leave this as an
  isolated experimental tool this turn; no GUI/default/field generality claims.
- This is retrospective source validation, not a fresh blind holdout. Model is
  limited to the original Dell scene; generic spatial guard rules do not make
  the trained representation generic. Existing source labels remain untouched.
