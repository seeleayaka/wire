# Frozen fine-CNN experiment (written before validation inference)

One new configuration only: input max-edge 784; grid 42x56; same frozen generic
YOLOv8s-seg prefix, layers 2/4, 3x3 pooling, per-layer L2, RGB/255. No training
or downloads. 80 fit / 10 embargo / 30 calibration train normals are copied
exactly from the current CNN snapshot. Same 32 selected channels and unweighted
position mean; local 3NN radius 2 preserves approximately the coarse radius-1
physical tolerance. Calibration scales and full-map-max p95 use train normals only.

All 30 val01 images, no test01. Same seed-0 registration and canonical normal_073
ROI. Recompute 392/21x28 features on the exact same aligned pixels, compare against
all 140 old feature caches and record maximum differences. A mismatch means this
is not an isolated resolution comparison and must be disclosed/rejected.

Endpoint A: fixed old anchor, qualified support rule and matched old candidate
count, fine score replaces coarse score. Compare frozen qualified baseline and
old DINO baseline. No new candidate geometry generated at this endpoint.

Endpoint B: preserve frozen qualified parents; apply the existing one-focus-hint
rule, allowing only the new map shape. All geometric/support/area rules unchanged;
fine train threshold replaces coarse train threshold. Report hint-only precise
IoU and added fault/normal region burden separately. Do not inflate parent+hint
union into an accuracy claim. Parent preservation cannot prove human utility.

Gate: no promotion if field-scene evidence is absent; validation improvement must
not hide per-kind regressions or normal burden. If resolution does not produce
precise source matches, reject this route as tested instead of searching settings.
Record registration quality, source/weight/library/code hashes, per-stage time,
peak observed RSS, features/maps, calibrated threshold and all 30 case outcomes.
