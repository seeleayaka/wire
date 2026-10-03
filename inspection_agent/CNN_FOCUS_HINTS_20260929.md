# Parent-preserving focus hints: limited localization utility, not promoted

All 30 fixed val01 traces only. No test01 access, model training, download,
threshold search, per-image rules, default CLI or GUI modification.
The brainstorming scope remains existing-dataset review with general rules;
diagnose drove fixed-policy comparison and separate utility/burden accounting.

Rule fixed before evaluation: at most one hint per parent, from non-edge raw
tile/refinement observations; area <= half parent; at least one different source
tile with IoU>=.25; CNN region p95 strictly above frozen normal-map-max p95
3.598051297664642. Rank by number of distinct supporting tiles then CNN p95.
Missing eligible support abstains. Original parent dictionaries stay unchanged.
Hint role is `within_parent_manual_review_hint`, automatic_fault_verdict=false.
No labels enter hint selection. Source labels are used only for the metrics.

## Results

- Parent counts/geometry and metrics exact: 70 fault/37 normal candidates,
  164/245 overlapping source fragments, 15/15 fault-image overlap, 4 IoU>=.5.
- Added hints: 16 on 12/15 fault images; 0 on 15 normal images.
- Hint-only overlap: 32/245 fragments across 10/15 fault images. Nine fragments
  have IoU>=.1; zero have IoU>=.5. Four IoU>=.1 fragments are new beyond parents.
- Hints overlap a source target in 14/16 cases, but coarse intersection is not
  precise localization, and incomplete labels prevent field-precision claims.
- Combined parent+hint metrics are deliberately not presented as improved
  accuracy. Parent coverage is unchanged by construction, while fault review
  regions increase from 70 to 86 (+22.9%). Human benefit/time is not tested.
- By kind hint count: damaged 6, disconnected 3, misrouted 7.

No evidence of improved precise localization. Keep this as a diagnostic prototype;
do not promote it into the GUI or default-off CNN runtime.

## Spatial-resolution diagnosis

Frozen ROI 3430x2518, CNN map 28x21: approximately 122.5x119.9 source pixels/cell.
Source target median width/height: damaged 41x38, disconnected 119x89,
misrouted 219x197 pixels. Damage fragments are smaller than one current cell.
The 392-max-edge CNN input reduces median damage width/height to about 4.7x4.3
pixels before network downsampling. This supports, but does not prove, an upstream
resolution limit. Source fragments are not complete electrical cable objects.

Fixed smallest filename per kind panels were rendered and inspected:
damaged_007, disconnected_002, misrouted_012, normal_001. Yellow parents, cyan
hints, red scoring-only source labels. Fresh visual registration is separate
from frozen metric computation. Hints can still target harness/support structures;
neither heat strength nor cross-tile agreement proves a fault.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -B tools/evaluate_focus_hints.py --output output/fresh_hint_run/report.json
.\.venv\Scripts\python.exe -B tools/render_focus_hints.py --report output/fresh_hint_run/report.json --output output/fresh_hint_run/panels
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_focus_hint.py
```

The pure hint helper has six tests: no mutation/aliasing, same-source duplicates
cannot corroborate, strict threshold, edge/large-child rejection, invalid input
rejection, and empty-pool abstention. Cached repeat reproduces the report exactly.
Evidence is `output/cnn_focus_hints_val01_20260929/`.

Next safe experiment: higher-spatial-resolution CNN evidence on fixed train/val,
preserving the existing normal-bank split and paired registration. Freeze one
resolution/feature rule ahead of evaluation; measure local-bank runtime/memory,
precise IoU, source coverage and normal burden. Compare within-parent evidence
before altering production boxes. Prior DINO heat/high-resolution failures remain
negative controls, not proof this different CNN route will work. Do not tune test01.
