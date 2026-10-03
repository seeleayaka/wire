# Next bounded crop-training protocol (training not started)

Frozen from train01 audit, before new training. Never use outer val01/test01 to
choose crop coordinates, training samples, weight epochs or confidence thresholds.

- Source train01 only; stable SHA-based per-kind source-image split: 192 inner
  train, 48 inner val. All tiles of a source image stay together. Exact image
  duplicates absent, but same chassis/camera and correlated capture times remain.
  This is not a device/session-independent holdout.
- Same generic 1280 squares, stride960, end-anchored; no fixed port locations.
  For inner training keep all positive tiles including every clipped rectangle
  intersection; plus at most one deterministic port-negative tile/source. Do
  not erase cut labels or discard all tiles intersecting a cut port. Result:469
  train crops (278 positive,191 negative);921 rectangle labels (203 clipped).
- Inner validation uses all12 tiles/source,576 crops; no GT-based crop selection.
  Evaluate detections after source-coordinate merge and the existing artificial
  edge guard, against complete source rectangles, not clipped-target mAP alone.
- Existing verified local generic yolov8s-seg.pt initializes training, NOT the
  earlier validation-selected Dell best.pt. The new YAML contains no outer val
  or test path. Rectangles are supervision for local cues, not actual masks.
- First bounded new run: CPU4, imgsz960, batch1, workers0, freeze10, SGD lr0.002,
  six epochs, warmup_epochs1, seed20260929; patience6, no mosaic/rotations/flips,
  translate .02, scale .05, amp/cache false. Pick best only on inner val.
  One fixed recipe, no hyperparameter sweep. Check source loss/gradients and RAM
  on an initial small smoke batch before committing to the full run.
- Train-normal calibration uses ONLY the24 inner-val normal source images, with
  all12 tiles and fixed edge rejection/class-awareNMS. Threshold=max(.25,p95 of
  their per-source maximum confidence), frozen before any outer val inference.
  This is an internal development holdout, not an independent field calibration.
- Before expensive full run: verify materialized crop/source hash manifests and
  model initialization file; read this protocol and crop audit. Keep prior
  weights/outputs and current formal path unchanged.
- Only after training/calibration freeze, evaluate all30 outer val images once,
  preserving existing parents and accepted whole-frame hint. At most1 new hint
  per image using unchanged containment/validity/dedupe rules. Compare5 precise,
  164/245 overlap,71 fault and37 normal regions. Accept only net precise gain
  without normal-hint increase; never lower thresholds to rescue a failure.
- test01 is not a tuning set. No cable identity/continuity/fault-verdict or
  cross-device accuracy claim. New training and this acceptance evaluation remain
  unperformed; materializing crops is not proof of improved detection.
