# Fixed 784 CNN resolution experiment rejected

Protocol `CNN_FINE_PROTOCOL_20260929.md` was written before inference. One fixed
configuration, no setting search: 784 max edge / 42x56 grid / radius 2 versus
392 / 21x28 / radius 1. Same prefix weights/layers, pooling/normalization, 80 fit,
10 embargo, 30 calibration train-normal split, seed-0 paired registration and ROI.
No test01, training, downloads, default CLI or GUI changes.

## Controls and reproducibility

140 coarse feature controls (80 fit, 30 calibration, 30 validation) are exactly
equal to the existing frozen caches: maximum absolute drift zero. All 140 global
registration quality gates passed. Source pixels, cache hashes, weights/library
versions, registration code, and original candidate code were checked.
This removes observed coarse-control drift as an explanation; it does not prove
registration is physically perfect or this chassis represents new scenes.

The pre-normalized local-bank implementation was checked against the original
distance function on real first-calibration data and synthetic border/radius/k
cases. Two new distance tests plus seven focus-helper tests passed, with 19 prior
CNN tests also passing. Coarse hint defaults still accept only 21x28; fine geometry
requires explicit `expected_grid=(42,56)`.

After inference, all 30 validation score maps, selected dictionaries and hint
lists were recomputed from fingerprinted feature caches and matched exactly.
Rebuilt spatial means matched exactly; the original train split was verified.

## Results, same candidate slots

| Metric | Frozen qualified coarse | Fine configuration |
| --- | ---: | ---: |
| Source fragments with any overlap | 164/245 | 151/245 |
| Fault images with any overlap | 15/15 | 15/15 |
| Image-equal overlap fraction | 70.07% | 67.43% |
| Source fragments IoU >= .1 | 25 | 24 |
| Source fragments IoU >= .5 | 4 | 4 |
| Mean best source IoU | .03758 | .03383 |
| Fault candidates | 70 | 70 |
| Normal forced-localization candidates | 37 | 37 |

Per image: 0 improved, 10 unchanged, 5 regressed on source overlap. Damage 91/143
to 81/143; disconnected 59/83 to 56/83; misrouted unchanged 14/19. Damage remains
zero precise source matches. All 15 normal geometric sets unchanged. Mean fault
union ROI area .20410 -> .19653; normal remains .20195. These are candidate/label
metrics, not field accuracy or the full upstream normal-bank system false alarm rate.

## Why no hints

Fine train-normal full-map-max p95 threshold is 6.940047932. Coarse was 3.598051298;
these are separately normalized feature spaces, not directly comparable raw scores.
The fine rule yielded zero hints on all 30 val images and zero additional precise
fragments. A label-free gate audit found 312 non-edge, half-area, spatially
corroborated fault-group members before the CNN threshold, but their maximum p95
was 6.803386331; none passed. Normal-group pre-gate members 139, max 2.845342469.
This is calibrated abstention, not missing hint wiring. Fine qualified-support
exception contributed just 1 extra selected fault candidate and 0 normals.
The audit did not lower or search thresholds.

## Artifacts, cost and decision

`output/cnn_fine_validation_20260929/`: completed report, 140 feature caches,
30 fine maps, calibration/model files, progress checkpoints, paired_summary,
replay_verification, gate_audit and fixed smallest-filename visual panels.
damaged_007, disconnected_002, misrouted_012, normal_001 panels were inspected.
The wider harness/chassis candidates remain; finer ranking did not solve precision.

Full bank/calibration/validation preparation took 802.7 seconds (~13.4 min).
Mean fresh registration 4.94s/image; fine feature extraction .084s/image; local and
position scoring 1.19s/scored image. Normalized bank 137.81 MiB, maximum observed
RSS .578 GiB. RSS is sampled, not guaranteed true peak. Cached DINO candidate
traces were reused; this is NOT live GUI or complete DINO/SAM pipeline latency.

Reject both fine re-ranking and fine hints for promotion. The active default-off
CNN branch stays frozen at 4bd13c2; no algorithm deployment improvement claimed.
Do not conclude all high-resolution CNN methods fail, nor that increasing resolution
alone can rescue fine electrical faults. Generic normal novelty and the existing
candidate geometry remain limits, with incomplete fragmented labels.

Next safe direction: add localized structural/wire-segmentation evidence inside
the preserved candidate regions, with reliability/abstention rules. First inspect
existing SAM/visible-wire coverage and identify a reproducible seam; do not assume
wire masks establish electrical identity, connectivity or fault class. Continue
train/val-only measurement and preserve old candidate/normal burden until a net
gain exists. This changes evidence, instead of another threshold/resolution sweep.
