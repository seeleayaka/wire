# Candidate-footprint frozen visual descriptor

Actual box-aspect audit shows1032 exactly identical current crops with666
contradictory TRAINIoU labels. Linear shape append failedP89.72%/GTrecall7.85;
one fixed nonlinear32-GELU shape/context control yieldedP94.67%/GTrecall73.55,
still below unchangedP98 gate; both rejected, no source/heldout claims.

Change only descriptor: frozenDINO cropCLS plus area-weighted patch mean
INSIDE actual candidate footprint replaces fixed central4x4 patch mean.
Actual floor/ceil crop extent maps box to16x16 token grid; partial boundary
patches contribute fractional intersection area. Retain same1.5/3contexts,
1536 observed/expected vectors and6144 paired channels; no coordinate or name
features, no encoder training. Same-crop forward cache may reuse ONLY exact
crop signature, pool each candidate independently with its footprint.

Use same3134 original geometry training rows plus1032 aspect training rows
from rejected shape probe (4166total), same source folds/labels/344GT. Normal
reference-self rows must use expected/expected, not inspection/expected.
Reuse original verifiedSIFT matrices/context-valid sample selection, not GT
alignment. New descriptor requires fresh frozen encoder forwarding; cache
pins/classifier/label/fold/model/source/reference SHA protect preparation.

First software geometry/dedup/reference-self tests, then smoke first2 eligible
TRAIN sources, checkfinite and real forward counts/checksum/pooling footprint
sensitivity. Only then full192 extraction, one process,2CPUthreads; no SAM
concurrent model. Fixed original400-step linear head/AdamW.01/.001/seed0/class
weights/OOF3/probability.98/P.98/GTrecall.25,last checkpoint, no tuning.

After crop pass, actualTRAIN192 novel candidate OOF additive gate against
accepted geometry286/344/4 prefix,5+5, requireTPgain/no extra unmatched/no
old target loss/normal0. Only after source pass fit fixed full head then
fresh48inner vs64/80/0 and30outer vs33/56/1. No epoch/score/GT edits or photo
special cases. Validation reused, not field or cross-cabinet proof. Still
needs actual new reference/ROI/SAM/Qt if all source gates pass. No automatic
deployment and no changes to current accepted release.
