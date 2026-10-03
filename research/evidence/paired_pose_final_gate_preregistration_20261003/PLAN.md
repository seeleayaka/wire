# Evaluate the unchanged complete safety gates, not raw proposal FP alone

Raw uniform-pose proposal search291/6 vs289/4 rejects; preserve that result.
New falsifiable hypothesis: original reference/ROI/warp98/duplicate/budget
filters may remove false extra source proposals without removing true gains.
Do not change a filter, score.98,model/vote threshold,sourcepose generation,
actual native-hint linking or any existing accepted median cue.

TRAIN: ALL FOUR prospective added-cue cohorts (not only two gains), selected
from already frozen raw source report, plusfirstnormal from each group. Fresh
original InitialReviewWorker SIFT/DINO -> actual accepted E median review ->
workspace pose proposal adapter through unchanged original reference whole
plus640/960 teacher/student views/ROI/warp98 gates. Source/model/head/code SHA
and fresh actual native prefix/proposal parity required. No GT input to policy.
All192 unchanged no-new-cue cases remain exact frozen source prefix; aggregate
newonly accepted native cues linked from final hints, scoring AFTER final gate.
Strict TPgain vs289/344/4, no newunmatched/oldloss/normalcue. Explicit localECC
safety controls accepted as abstention only, never recognition success.

On failure STOP. On pass new paired inference on ALL48inner native poses and
ALL30outer, then ALL their prospective newcue cases through fresh unchanged
final safety gates, innerstrictgain>65/80/0,outer>=33/56/1. Allprevious cues
preserved; complete newSAM/actualQt and artifact hashes needed before release.
No automatic integration, labels/gates/budgets altered, no field claim.
Fine-tile job continues unchanged independently; CPU/memory bounded, no SAM
concurrent with both tasks and no user app/window closures.
