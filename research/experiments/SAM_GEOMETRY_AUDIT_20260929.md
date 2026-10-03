# SAM geometry guard and fixed validation pilot

## Decision

Implement the endpoint ambiguity guard. Do **not** deploy SAM component bounds
or parent-mask intersections as a precision improvement. The coarse experimental
CNN branch stays at its frozen behavior; the GUI and dirty SAM fusion are untouched.

## Implemented change

The old extractor silently selected the farthest pair from any multi-tip
component. The new extractor preserves all candidate tips and only emits a
visible pair for one connected skeleton with two tips and no branch clusters.
Loops, branches and disconnected skeletons abstain, without spur pruning or
scene-specific thresholds. This is visible geometry, not electrical connectivity.

The adapter also rejects old reports whose candidate_tip_count reveals collapsed
ambiguity, rejects branch metadata and incomplete/inconsistent version-1 geometry.
Legacy reports without geometry metadata preserve the earlier controlled contract
but are explicitly marked geometry-unverified. **Regenerate old endpoint reports
from masks** before using them as geometry-checked evidence; the metadata cannot
retroactively detect every hidden branch in an old two-tip report.

## Evidence

- Three existing train01 cable/0.4 caches: normal_003 23 components/16 pairs,
  damaged_001 16/13, disconnected_001 62/37. Total 101 components: 66 eligible,
  35 abstentions. Cached masks are internally consistent; historical source and
  checkpoint hashes were absent, so current hashes cannot prove historic identity.
- Fresh val01 disconnected_002: 10 SAM instances, 15 source components, 12 pairs,
  3 abstentions. After perspective resampling, only 10 of 15 skeletons pass the
  no-branch check: pixel-grid branches can conservatively reject a simple wire.
  No attempt was made to prune branches to fit this example.
- 11 new geometry/adapter tests and 7 previous adapter tests pass. The actual
  endpoint CLI was run on all four images; cached scoring/registration replay is
  exact. Without declared terminal ROIs, none of these reports emits an edge.
- Model build 9.274 s, image encoder 242.567 s, text/masks 18.898 s, summed
  270.739 s excluding process startup and output serialization. Fresh encoder,
  production local checkpoint, CPU 8 threads, resolution 1008, cable, threshold .4.

## Pilot localization diagnostic (13 fragmented source rectangles)

All masks and annotations were mapped with the actual production SIFT homography
(seed 0, temporarily captured and restored). Transformed scoring rectangles
match the frozen val cache exactly; masks use nearest-neighbor warp and valid
source coverage. No labels are used to generate/select boxes.

| Diagnostic regions | Count | Any bbox overlap | IoU >= .1 | IoU >= .5 |
| --- | ---: | ---: | ---: | ---: |
| Frozen coarse parents | 6 | 11/13 | 2/13 | 0/13 |
| All SAM component bounds | 15 | 10/13 | 4/13 | 0/13 |
| All parent-mask component intersections | 11 | 9/13 | 3/13 | 0/13 |
| Parents plus all intersections | 17 | 11/13 | 3/13 | 0/13 |

Raw mask pixels intersect 7/13 source rectangles. These annotations are not wire
segmentation truth, so this is **not** SAM mask recall. Adding all 11 intersections
keeps coarse overlap but increases review burden without a precise hit. Whole-wire
bounds can improve moderate geometric proximity without locating the actual fault.
Every SAM region is retained for a ceiling diagnostic: no ranked detector, no
budget-matched gain, no normal false-positive estimate, no dataset-wide precision
or new-scene claim. One validation image cannot invalidate every SAM-based method.

## Artifacts and next safe entry point

- `output/sam_cached_geometry_audit_20260929`: three-cache audit and inspected board.
- `output/sam_geometry_guard_20260929`: actual endpoint CLI reports and review sheets.
- `output/sam_fixed_val_pilot_20260929`: fresh masks/state, final_audit,
  final_replay and final_verification.json. Original source/checkpoint/runner fingerprints
  are recorded by the audit, not present in the legacy runner's report schema.
- Reproduce using `tools/audit_sam_fixed_pilot.py` and
  `tools/verify_sam_geometry_audit.py`; reuse masks, do not rerun the encoder.

Next precision hypothesis should distinguish **a changed local connector/terminal
state from a whole visible wire**. First establish train-only reliable connector
regions/evidence with fixed rules, then freeze and evaluate the complete existing
validation set and normal review burden before integration. This is a next
hypothesis, not implemented or validated. Do not lower thresholds or crop to source
annotations on this pilot. The topology guard is a completed reliability repair,
not completion of the user's precision/recall objective.
