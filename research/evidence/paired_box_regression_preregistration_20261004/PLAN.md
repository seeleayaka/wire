# One relative box-regression TRAIN geometry prototype

Small nonlinear semantics failed297/12 versus295/4; do not repeat capacity or
confidence sweeps. Separate localization from semantics: one zero-initialized
6144->4 linear SmoothL1 head on audited existing positive TRAIN paired-feature
examples. Relative center offsets and log width/height, not absolute locations,
source names, metadata or template GT in inference. Formula inspiration:
https://arxiv.org/html/1506.01497v3 (relative anchor box regression). This is a
local hypothesis, NOT implementation of that full model or its reported gains.

Existing9808-feature corpus, only labels1/2 with same-class best TRAIN GT IoU>=.5
for regression loss; no negative regression labels/validation fitting. Keep
source modulo3 folds exactly, seed0,400 full-batch AdamW lr.01/decay.001. Fixed
relative inference/training clamps center[-.15,.15],width/height scale[.8,1.2],
same envelope as already enumerated generic poses. Same frozen DINO features.

Predict on ALL cached original native proposal features in all192 TRAIN images
using source-excluded heads. Keep original candidate pool and all accepted295
prefix cues; refined geometry must recompute3unique checkpoint votes at IoU.5,
original border16/context.85 and no overlap>=.5 with old accepted cues. Shared
extra budget must have room. No classifier score or confidence changes yet.

First report ONLY upper-bound weak-target proposal coverage among the49 missed
targets: old eligible proposal pool versus its union with refined boxes. Also
report positive-example OOF IoU and any worsening, never call union coverage
recognition accuracy. This is geometry feasibility, not final confidence,
normal-reference/ROI filtering or shared-budget allocation. No full head fitting
or heldouts at this stage. Only if geometry adds coverage without removing old
proposals consider a separately pinned original-pixel semantic/actual trial.
No E changes, model deployment, new photos, SAM or paid computation. RAM>=6GiB
at launch, CPU2threads, all feature/metadata/label/model/runtime SHA pinned.
