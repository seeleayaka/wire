# One fixed small nonlinear semantic head; TRAIN-only source gates

The previous linear fine-candidate head failed because extra unmatched cues
remained. Test a limited nonlinear boundary on the independently audited SAME
9808 TRAIN paired features (including normal-reference twins); no new images,
feature extractor, proposal rule, vote rule or GT changes. Exactly one fixed
6144->16 ReLU->dropout.1->3 head, seed0, AdamW lr.01/weight_decay.001,
400 full-batch steps, same class-balanced loss as the linear experiment.
No hyperparameter sweep, validation-label selection or classifier confidence
threshold changes. This is an architectural hypothesis, not accepted accuracy.

Reuse all969 actual fine native proposals across ALL192 original TRAIN sources,
original paired DINO features, baseline295/344 with4 unmatched. Three existing
source folds exclude every same-source example AND its reference-self twin.
Only this new classifier is OOF: existing detector and immutable prefix saw
TRAIN, and reference normal073 is shared conditioning, not independent data.
Same p>=.98,3distinct model SHA,7 relative poses,context>=.85,5+5 budget.

First source-classifier-OOF must strictly increase TP, no unmatched increase,
no old loss, zero normal additions. Otherwise stop without full head or heldouts.
Only if OOF passes fit ONE full head and repeat source gate; full pass still
requires fresh complete INNER/OUTER fine proposals + actual normal-reference/
ROI/SAM/Qt gates separately. No automatic deployment. Never change installed E
models/config/core or in-flight stress/proposal scripts. Cached tensors and
metadata/model/audit SHA pinned; at least6GiB RAM, CPU2threads, no GPU/cloud/SAM.
