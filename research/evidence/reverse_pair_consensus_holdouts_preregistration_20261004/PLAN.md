# Fixed INNER then conditional OUTER, combined reverse-pair geometric cascade

Audited source OOF297/4 and full298/4 vs installed295/4 passed. New classifier
was fitted only on TRAIN, including reverse positives as OTHER. No further fit.
INNER48 baseline installed68/80+0 unmatched, must gain and lose no prior targets.
OUTER30 baseline40/56+1 unmatched, no regression. Normal cue count must be0.

INNER uses the exactly pinned prior fresh teacher/student fine960/stride720
outputs from fine_consensus_rank_holdouts_20261004, not a new detector run.
Check source pixels, reference, weights, configuration, old runtime, alignment,
pool and proposal geometry. Re-extract ORIGINAL pixel DINO pairs for the new
head; verify old head probabilities on these same fresh features before new
scoring. Save fresh vectors and old-head consistency results per case.
Reused detections are disclosed; fresh scoring is not fresh detector inference.

If INNER passes, OUTER uses new detector and registration inference, because
the prior fine worker never reached OUTER. Fixed960/720/.001/.7/maxdet300 and
same edge filtering. No GT at inference, target labels used only after scoring.
No new thresholds, training, reference, budgets or E mainline modifications.
Holdout pass still requires actual reference/ROI/Qt/SAM acceptance before use.
This is repeatedly examined development data, not unseen-cabinet field accuracy.
