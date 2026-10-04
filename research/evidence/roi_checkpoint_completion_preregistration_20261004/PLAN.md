# Complete checkpoint coverage of an acquired ROI (contingent experiment)

Start ONLY after zoom2 finishes, is rejected and independently audited.
If zoom2 passes source, prioritize its required validation; do not launch this.
Do not compete with active model inference. This document is fixed before
the outcome of zoom2, and contains no validation-derived parameters.

ALL192 old evidence-first acquisition failed298/4 ->298/6,normal1. Its
independent ALL46 miss analysis found:30budget-full; of16with room,9not
covered by any acquiredROI,5covered but not by all missing checkpoints,
2still failed despite all missing checkpoints queried. No raw-vote increase.
This is a TRAIN-only diagnostic, not proof of field accuracy or a runtime
GT rule. The next specific hypothesis is checkpoint routing: anROI can
contain objects other than the seed, so calling only the seed's one missing
checkpoint may leave nearby objects without all3independent detector votes.

Change only which checkpoints observe an already chosen960ROI. Keep EXACT
old157 geometry-only actions,89sources, same960window offsets+-120 and
input960. For each action query all3existing checkpoint SHAs over the same
two windows. Reuse the old audited missing-checkpoint outputs with exact
source/window/checkpoint/reportSHA matching; freshly infer other2checkpoints.
The two views of one checkpoint remain ONE vote. Do not add a fourth weight,
count views as votes, add action slots, change regions, or change final gates.

This deliberately returns to the original960 control to isolate checkpoint
coverage, NOT combine scale and routing effects or cherry-pick the better
scale. Old zoom2 is evidence for one rejected independent hypothesis only.
The same original298/4research prefix is protected; failed trial additions
are not inherited. Final original-pixel DINO9459 head,p.98,3SHA,.85support,
IoMin.5 and5+5 unchanged. No GT/names/absolute positions as inference rules.

Preflight generic query/source/weight/immutability tests, action identity
ALL192 and imports. One2CPU-thread worker, finite7200sec; separate bounded
audit watcher. Cache pins and original/runtime/CONFIG checks protect inputs.
Auditor independently enumerates3checkpoint requests per action, verifies
reused evidence equality and all fresh coordinate bounds, replays SHA-unique
votes, original head probabilities from savedfeatures, rank and ALL192GT.
It does not rerun detector inference or independently re-extract DINO pixels.

Require sourceTP>298,unmatched<=4,no old loss,normal0. Failure stops validation,
preserves result and must not trigger a threshold or action-budget sweep.
On sourcePASS, fresh combined INNER>68/80+0; OUTER>=40/56+<=1 and all actual
reference/ROI/Qt/SAM gates required before deployment. No private image
publication,paid compute,new download or mainline modification. No claim of
cross-cabinet accuracy from development splits or diagnostic upper bounds.
