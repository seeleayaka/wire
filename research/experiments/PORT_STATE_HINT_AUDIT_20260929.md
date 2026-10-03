# Existing supervised port cue: small positive localization gain

## Decision

The frozen within-parent hint experiment passes its limited validation gate:
one additional precisely matched source rectangle, no new normal hints, all
coarse parents preserved. Keep it as an isolated optional experimental tool;
do not change the GUI, the coarse CNN CLI or the dirty SAM fusion implementation.
This is not completion of the overall precision/recall objective.

## Fixed experiment and limits

- Reused the old 3-epoch, 960px YOLO rectangular-segmentation checkpoint for
  unplugged_plug/unplugged_jack. No retraining or download, no coordinate recipe.
- Operating threshold .25, fixed from max(.25, train-normal maxima p95). The
  30 training normals were already seen by this supervised model. The checkpoint
  was originally validation-selected. This is retrospective source validation,
  **not** new blind validation, independent calibration or cross-device accuracy.
- Original full-frame predictions, then predicted boxes and scoring rectangles mapped with the
  actual production SIFT homography (seed0). All 30 scoring rectangle lists match
  the frozen val cache exactly; 60 source-file fingerprints and model/code hashes
  verified. Source annotations are scoring-only inputs after hint selection.
- Hint rules: score above threshold, >=.98 valid source coverage, entirely within
  a parent, <=half parent area, at most one per parent, confidence-only stable
  ordering. Parents keep exact dictionaries, geometry, metadata and order.

## Full fixed validation result

| Metric | Frozen coarse parents | Parents plus port hint |
| --- | ---: | ---: |
| Fault images with any annotated overlap | 15/15 | 15/15 |
| Fragment rectangles with any bbox overlap | 164/245 | 164/245 |
| Fragment rectangles with IoU >= .1 | 25/245 | 25/245 |
| Fragment rectangles with IoU >= .5 | 4/245 | 5/245 |
| Mean best fragment IoU | .037577 | .040565 |
| Fault review regions | 70 | 71 |
| Normal review regions | 37 | 37 |
| Added normal hints | 0 | 0 |

The single accepted hint is on disconnected_027 and has a class-consistent precise
match. No damage or misrouting precision gain occurred. There are 56 class-3/4
source rectangles in this fixed validation set: the hint has 1 matched rectangle,
0 unmatched hints and 55 missed rectangles. **1/1 matching is not 100% system
accuracy**; it is one prediction. Existing 37 normal candidate regions remain.

## Why only one hint, and next hypothesis

Cached predictions contain only two valid boxes above the unchanged threshold
across all 30 validation images. One is eligible inside a parent; the second,
on disconnected_002, crosses its parent boundary and is rejected. The detector
therefore has substantial missing local evidence; simply changing attachment
rules cannot fix most missing port targets.

Diagnostic ablation with **all** two fixed-threshold port boxes gives 6 precise
rectangles vs 4 parents-only, fault regions 72 vs 70, normals still 37. Confidence-
only top1/image gives the same result because each image has at most one such
prediction. This is a cached retrospective diagnostic, not a separately frozen
and independently validated new integration. No threshold was reduced to obtain
these results; do not turn the second picture into a custom clipping exception.

Next safe work: establish a generic object-evidence channel that preserves whole
local detections instead of forcing them inside coarse difference rectangles,
with explicit review budget and normal controls. Separately improve small-object
port evidence using train-only development and then freeze a validation protocol.
Do not retrain/select against test01, reuse ground-truth boxes as inference crops,
claim full seating/continuity, or assume the old Dell-only detector works in cabinets.

## Verification and artifacts

- 27 tests passed: 9 hint tests (duplicate ordering, no mutation, shape/warp/
  threshold rejection), 11 SAM geometry tests and 7 previous adapter tests.
- All 30 hint selections replay exactly, including reversed prediction order;
  training threshold rebuild exact; all 60 source-file hashes match.
- Actual experiment elapsed 186.0 s for 60 detector passes and 30 registrations,
  plus report/panel work. DINO/SAM were cached, so this is not GUI latency.
- All above-threshold images rendered without label-based image selection:
  disconnected_002 (magenta rejected) and disconnected_027 (cyan accepted), with
  frozen parents and scoring annotations. Fixed first-name damage, routing and
  normal control panels were also generated.
- `output/port_state_hints_validation_20260929`: report, calibration, prediction
  cache, replay verification, gate audit and panels. Reproduction tools:
  `evaluate_port_state_hints.py`, `verify_port_state_hints.py`,
  `audit_port_state_hint_gates.py`, `render_port_state_hint_panels.py`.
- First launch failed before inference because an absent per-run config folder
  made Ultralytics try a denied fallback. Corrected by creating that exact folder
  before import; no global settings changed. The failed workspace output is kept
  separate and is not a scored experiment.
