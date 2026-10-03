# Stricter model support for new searched boxes only

After TRAIN failed maximum-pose search291/344 with6 unmatched vs289/4,
the independent audit shows two gain cues have3 distinct checkpoint votes
and both extra cues have2. Test a general stricter support hypothesis, not
a per-case repair: EACH newly searched cue requires >=3 distinct available
detector checkpoint SHAs on its actual box (sameweightviews count once).
Do not change any old accepted cue/two-vote median policy or original head,
probability.98,geometryIoU.5/context85%,oneparentmax/shared5+5/labels.

Predeclared trial: all192 frozen TRAIN pose proposals/probabilities reused
with full SHA and currenthead identity; selection finalized before unchangedGT
scoring. If strict TRAIN gain with unmatched<=4/lost0/normal0, new DINO paired
inference on already frozen210inner and209outer poses, same actual registration
or verified source/reference cache as original gate. Strict inner improvement
over65/80 with0unmatched; outer>=33/56 with<=1 unmatched. Stop first failure.
No fullhead fitting or deployment. All validation has been reused already;
confirmation does not prove cross-cabinet generalization/field accuracy.
Full reference/ROI/nativehint/live/SAM/Qt still required before integration.
