# Observed-versus-reference semantics, 2026-10-03

The joint-quality locator passed crop precision91.5% but was rigorously rejected
after completed source outputs proved a new unmatched cue. No detector/head
deployed. New hypothesis: a real normal connector can look like the target
object yet be unchanged; source-only object classification is insufficient.

Keep existing proposal rule: TWO distinct teacher/first-student checkpoint
votes, confidence>0.05, same class IoU0.5, complete16px frame margin. This
existing proposal floor is not a fault confidence or deployed threshold change.
No partially inferred feature peer. Fixed camera normal073 expected reference.
Raw-candidate geometry upper bound (not achieved performance) permits gains
train19/inner6/outer10 under old extra budget; do not use this oracle at runtime.

Extract frozen DINO224 source crop embeddings at existing1.5x/3x context scales,
and matching expected-reference crops warped back into SOURCE coordinates by
the original mainline reliable full-frame SIFT homography. Reuse hash-verified
existing alignments when available; otherwise original frozen registration.
Cache identical reference SIFT descriptors without changing its algorithm.
Require minimum context valid-warp coverage0.85; uncertain source registration
and invalid crops are explicit abstentions, never clean negative observations.

Train-only192 source groups. Features concatenate source,expected,absolute
difference and elementwise product (4x1536=6144); no source names, labels,
coordinates, transforms, scores or metadata as model features. Classes other,
unplugged plug,unplugged jack. All original GT3/4 positives; matched reference
self-pairs are normal examples; up to2 existing training other/grid examples
per source; all novel two-model proposals labeled by original same-class IoU0.5
for hard semantic/geometry negatives. No annotation changes. These extra
classifier labels are derived training supervision, not new human annotations.

Three source-group folds by sorted192 source index modulo3. Fixed linear head
400 AdamW steps lr0.01/decay0.001/inverse class weights, same as old crop
classifier. OOF refers ONLY to this small classifier: YOLO proposals were
trained on192 sources, and expected normal reference is shared by design.
GT-crop feasibility precision98%/recall25%, probability0.98, fixed settings.
Include all negative/reference-self/hard-proposal examples in precision errors.
No best iteration/score tuning. If crop feasible, OOF actual ALL192 proposal
source gate (net gain,no unmatched/old loss,normal0,old5+5) before training a
full head; then inner48 and outer30, never used for head training.

No candidate confidence rewrite without provenance: retain detector proposal
score separately, and report effective paired classifier probability. Require
predicted class equal the two-detector proposal class. No auto-deployment;
source pass still needs current real source/reference ROI and Qt/SAM acceptance.
Same-camera reused evaluation only; not field faults,electrical continuity or
cross-cabinet generalization. All original model/core/calibration/GT unchanged.
