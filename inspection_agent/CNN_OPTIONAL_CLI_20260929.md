# Default-off CNN single-image review, 2026-09-29

The frozen CNN support exception now has a separate executable JSON CLI. The old
CLI, GUI, normal-bank gate, registration and DINO candidate rules are unchanged.
Only explicitly passing `--experimental-cnn` enables the additional ranking.
This is manual-review localization, not cable identity, electrical continuity,
automatic fault classification or validated field accuracy.

## Run from E:\PythonProject10

```powershell
.\.venv\Scripts\python.exe -B tools/run_mendeley_cnn_experimental_review.py --image "data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images/val01/disconnected_011.JPG" --model output/mendeley_normal_bank_20260926_v1/normal_bank_model.npz --train-normal-dir "data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images/train01" --experimental-cnn --output output/my_fresh_run/review.json
```

Omit `--experimental-cnn` for the old candidate behavior. Use a fresh output path;
existing reports are not overwritten. Default snapshot is
`output/mendeley_cnn_heat_evidence_20260929`. This snapshot only supports the
frozen normal_073 reference pixels/ROI, fixed-reference mode and budget 6.
Other contexts, missing weights/cache, runtime/source fingerprint mismatches,
invalid score maps or unsupported capture paths retain old candidates and record
`experimental_cnn.status = fallback` plus a reason. Normal-like images and failed
upstream registration do not load the CNN. Empty old candidates stay empty.
An old-pipeline exception still propagates rather than masquerading as success.

The adapter temporarily captures internal functions and restores them in finally.
Use it as a standalone single-image CLI only: it is not thread-safe, not a GUI
adapter, and does not publish updated overlays. All added code lives in separate
files; no old production algorithm file was edited. No model training/download.

## Verified

- 19 CNN-focused unit tests passed (14 test_cnn*, 5 experimental_cnn).
- 23 original regression tests passed (16 tiled, 3 anchored, 4 normal-bank).
- 60 immutable cached cases (30 val01, 30 previously completed test01) exactly
  reproduce the frozen qualified candidates; repeat/off/injected-failure checks
  pass for all 60. This is implementation regression, not a second test experiment.
- Actual val01 disconnected_011: upstream suspicious, CNN applied, 6 candidates,
  geometry exactly matches the frozen validation result; threshold unchanged at
  3.598051297664642. Normal_001: upstream normal-like, 0 candidates, CNN skipped.
- Evidence: `output/cnn_cli_smoke_20260929/` with two live reports and
  `replay_verification.json`. No test01 thresholds or strategies were changed.
- Replay-check development initially confused test trace source_sha256 (image
  bytes) with val trace source_sha256 (candidate code). Corrected the verifier to
  check each schema against its actual source; did not alter historical reports.

## Artifact restoration

Quota-related write approval failure is historical and now resolved. The full
test01 output and fixed visual panels were restored to the original project under
`output/mendeley_cnn_frozen_test01_20260929` and
`output/mendeley_cnn_frozen_test01_panels_20260929`; report hashes matched.
The original acceptance scripts are archived without modification under
`inspection_agent/reproductions/CNN_TEST01_20260929/`. Their historical output
guard is script-directory-relative; they are evidence, not instructions to repeat
test01 or tune against it. Old audit/protocol notes retain their historical context.

The completed test acceptance remains 163/242 -> 181/242 overlapping source boxes,
with normal forced-localization candidate geometry unchanged. IoU>=0.5 is only
2 -> 3; damaged-wire precise localization is still zero. The same chassis, prior
test baseline exposure and fragmented annotations limit generalization claims.
Reference-bank sensitivity remains unresolved. Do not enable GUI defaults yet.

Next: assess precise localization without enlarging boxes, and review usability
on validation only with the same candidate budget and normal burden controls.
