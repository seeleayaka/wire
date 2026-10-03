# Equal distinct-weight coordinate median candidate geometry

TRAIN-only diagnostic first; no changes to accepted286/64/33 or normal
reference/ROI/threshold/budget/GT/physical-fault boundaries. Motivation:
current multi-model proposals select the highest-scoring existing native
box, not a geometric ensemble. WBF primary paper1910.13302 and implementation
https://github.com/ZFTurbo/Weighted-Boxes-Fusion inspired geometric fusion;
this control is NOT standard WBF: equal distinct-checkpoint coordinate median
avoids treating non-calibrated checkpoint confidences as common weights.

For each existing generic highest-score sorted weak seed, retain existing
same-class IoU0.5 supporting boxes with score>0.05 and full-frame16margin.
Choose highest-confidence one per distinct checkpoint SHA (different views
of one checkpoint count once). At least2 unique weights. Median four edges;
recheck full-frame16margin and at least2 original supporting checkpoints
same-class IoU0.5 against the resulting box. No transitive connected-component
merging, no angle/filename/position exceptions. Deterministic sorting/ties.
Append only candidates not IoU0.5-overlapping accepted prefix; preserve old
native proposal evidence and distinct-weight records. Existing5+5 budget.

First write TRAIN192 proposals WITHOUT GT, then read unchanged labels to
calculate maximum matching ceiling and missing-target funnel. This is ONLY
an upper bound/diagnostic, never runtime GT selection or achieved gain.
No holdout look if TRAIN offers no new useful localization or prospective
actual fixed-head path. If useful, actual fresh paired crop features with
fixed classifier98%/original gates must pass ALL192 actual source gain and
then fresh inner48/outer30 before any reference/ROI/SAM/Qt release. Any model
sharing/training augmentation is a separate preregistered comparison, not
a score sweep. Existing positive-jitter training is an independent run.
