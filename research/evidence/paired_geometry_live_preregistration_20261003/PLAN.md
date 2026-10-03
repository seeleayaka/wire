# Actual current-V3 workflow plus fixed geometry head acceptance

Uniform training geometry negatives passed fixed OOF classifier gate198
sampleTP/0 errors, GT193/344 recall56.10%; original344GT are unchanged.
Source OOF train277->281/344 with4 unmatched; fresh inner63->64/80 with0;
outer33/56 with1 unchanged. Exact FULL checkpoint238f517b... trained192:
full-head cached TRAIN192277->286/344,4 unchanged. No source head deployed.

Run fresh original whole-frame registration/DINO and actual currentV3 source/
reference port workflow on ALL full-head TRAIN candidate sources, ALL inner
candidate sources, ALL four old training cue controls, prior rejected paired
inner candidate sources and ALL rejected committee outer candidate sources.
Select cohorts by saved candidate/error records, not hard-coded names, and
persist categories. Add first normal control per split when not already
included. Source/label/reference/model/core/experiment SHA protected.

New branch computes exact original two-detector>.05 candidate proposals and
5+5 selection on fresh currentV3 evidence, paired frozen DINO1.5x/3x crops from
the new actual alignment, probability.98,valid reference coverage.85. Require
fresh native current/proposed geometry parity with fixed source preflight at
.001pixel and1e-6confidence. Source metadata never uses GT to select.

Then fresh teacher/first-student reference full and640/960 matched candidate
views; append through original append_verified_student/select_rescue using
RECORDED actual analysisROIs, valid warp.98,reference>.25/IoU.5 and shared5+5.
Exact mapped-hint native association, no inverse-warp scoring or GT oracle.
Do not change parents,existing cues,old rescue/supplementary prefix or decision.
Uncertainty/source identity errors fall back explicitly preserving old output.

Require zero old target loss or unmatched increase, normal0. Aggregate new
TRAIN and inner native gain after real reference/ROI gates must be positive;
outer negative cohorts no new error. Scope is diagnostic fresh workflow,
not new field validation. SAM remains pending and actual GUI enhancement
must stay blocked on SAM-pending reports. Pass still needs complete new
inspection SAM/Agent and actualQtbutton/callback with default-off policy.
No automatic deployment or claiming head classification as physical faults.
