# Frozen symmetric support plus real reference gating, 2026-10-03

The source-only symmetric student/feature graph was rejected: +5 labeled
hits but +1 unmatched on56 positive/control training images. It stays rejected.
This is a DIFFERENT pipeline experiment: the unchanged all-class graph followed
by the same current ROI, registration, warp and static-reference veto gates.
Neither source-only failure nor a narrow selected-case improvement is acceptance.

Freeze original graph candidate0.75, peer0.25, same-class IoU0.5 and two tiles
over0.5 or two strict rechecks over0.75, both source checkpoints unchanged.
Preserve Current V3 old source prefix and current real source-reference hints,
primary5 and all-extra5. No class filter, new score tuning or GT-driven regions.
Use existing dataset normal073 reference, same workflow ROI [.03,.04,.97,.96]
for all fresh diagnostic cases. Reference non-detection is NOT fault proof.

Enumerate ALL192 train candidates (including all normal/negative controls),
infer any previously placeholder feature peer whenever weak student support
could permit a strong peer. Pin all source/model/code/cached predictions.
Then run fresh InitialReviewWorker registration/DINO and current actual port
pipeline on EVERY source with graph additions, not only gains. Freshly infer
both source checkpoints and require exact graph native geometry/score/class
parity with the cached source selector. Both reference checkpoints run once
per process on normal073 plus each matched reference crop. Use the original
append_verified_student selection/safety gates. Preserve old hints verbatim.
Link accepted mapped hints to their exact original native boxes, then score
native boxes against original labels. No inverse-warp score improvement.

Stages: ALL192 training gate then48 inner then30 outer. Need net training
and inner hits, no unmatched increase/old target loss, normal new cues0;
outer nonregression. Any failure stops. A registration/provenance failure is
reported and excludes qualification, never silently counted as a clean zero.
Never auto-deploy. New formal GUI/SAM acceptance is a later requirement.
This reused same-camera benchmark is exploratory, not new independent or
cross-cabinet accuracy, electrical continuity or production fault diagnosis.
