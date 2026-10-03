# Separate duplicate novelty from semantic confidence

All9 added fine cues were reviewed with registered expected crops, not selected
success photos. Independent GT-link diagnosis: six lie over the closest target
already covered by the protected prefix; one new target is below IoU.5; two
are new matched targets. This is weak-label/visual evidence, not physical truth.
Original failed fine report stays rejected. The pixel-body/ring head is separately
rejected at OOF297/12 and is not used here.

Fixed geometry-only rule: intersection divided by the SMALLER box area >=.5
means duplicate for new additions versus ANY protected cue or already chosen
new cue. Use the existing half-overlap threshold, no sweep. Unlike IoU this
detects nested/different-scale duplicate boxes. No image names/GT at inference.
Never delete or rescore old cues. Test translation/scale/symmetry/nested/separate
and invalid input invariants first. This could suppress adjacent real objects;
the strict validation gates remain mandatory and current E remains untouched.

PhaseA: replay ALL192 original fine scores with ONLY this duplicate rule. Report
net TP/unmatched, no adoption unless all gates pass. A failed A does not authorize
weaker thresholds or changing GT. PhaseB is a distinct fixed representation:
reuse the previously trained relative source-OOF heads (6144->4,400seed0,
SmoothL1,.01/.001), without refitting, applied to ALL969 cached fine features.
Relative centre±.15/log-width-height .8..1.2. Recompute actual merged model votes
using full teacher/student/feature/alternative and real fine tile views; same
weight SHA remains one vote. Keep16border/.05model-floor/3SHA/.85context,
same old-prefix5+5 budget. Original+refined proposal coverage measured against
49 misses only as a diagnostic UPPER BOUND, not recognition accuracy.

Only if refined proposals add at least one previously uncovered missed target
to the eligible union, recompute ORIGINAL observed/expected DINO pixels and
the accepted9459 classifier at the final original+refined geometries. Cached
old semantic scores must not be attached to changed geometry. Same parent,
class and checkpoint vote gates; no full-box head/validation fitting. Entire
TRAIN must have TP>295,unmatched<=4,no old loss,zero normal cues before fresh
INNER strict gain,OUTER non-regression and actual reference/ROI/Qt/SAM gates.
If coverage fails, stop. If source classification fails, no held-outs or deploy.

Pin all source pixels, runtime, model/view/feature JSON/tensors, OOF regressors,
plan and scripts; audit independent geometric and selector/GT calculations.
No E modifications, full SAM recomputation, field or cross-cabinet claims.
