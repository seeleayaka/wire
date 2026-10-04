# Fixed raw-box consensus proposal experiment

Freeze accepted E native baseline:295/344,68/80,40/56,unmatched4/0/1.
Do not replace existing candidates, weights, labels or runtime source.
TRAIN-only remaining-miss audit found detector coverage omitted by the median
seed proposal path. Test one general alternative: generate the SAME seven
poses directly from raw model boxes, retaining three unique checkpoint votes.
Use unchanged frozen native classifier, p.98, valid context .85, existing
any-class IoU.5 dedup, one pose per overlapping raw seed cluster, shared5+5.
No source names, labels or absolute coordinates enter proposal generation.
No model fitting, sweeps, lower thresholds or photo-specific rules.
TRAIN192 must strictly increase hits, keep unmatched<=4, lose no old match,
and add no normal cue. If failed stop and preserve accepted runtime.
Only if passed run INNER48 with strict gain / unchanged unmatched, then OUTER30
with nonregression. Then independent audit, original reference/ROI gates and
actual workflow must pass before ANY E implementation. All developmental and
reused cohorts; neither source results nor candidate coverage are field accuracy.
