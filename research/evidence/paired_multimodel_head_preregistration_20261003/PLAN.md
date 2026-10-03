# Existing accepted head on expanded checkpoint-witness proposals

Ceiling audit not achieved accuracy: current286/64/33, old2model potential
296/69/43, expanded3distinctweights+1280view potential301/72/45. No new
accuracy claim; ceiling merely permits a finite actual classifier test.

Use EXACT accepted geometry checkpoint238f517b and original center4x4
paired descriptors6144/context1.5/3/.98/coverage.85. No training or score
tuning. The expanded deterministic2distinctweight weak proposals>.05/
sameclassIoU.5/frame16 come from ALL270 bound existing output records.
Same feature-weight at960/1280 does not count as two weights. Append after
current accepted geometry286/344/4,64/80/0,33/56/1 preserving old5+5/prefix.

TRAIN source/source-reference feature cache reused only with verified
source/ref/model/SIFT identity and EXACT both-scale crop signatures, excluding
reference_self samples. All remaining novel proposals get new frozenDINO
forwarding and stored evidence before GT source scoring. No synthetic failed
context placeholder as a negative. Unreliable registration/context abstains.
One CPU thread; separate footprint job2threads may run without shared writes.

Complete192TRAIN first: must gainTP, no unmatched increase/oldloss/normal cue.
Fail stops before holds. Then fresh48inner requireTPgain/no extra error and
30outer nonregression. Inputs/model/oldpolicy unchanged. Registration on all
holdout sources may reuse bound earlier original homographies; no GT-based
alignment. Validation reused, fixedhead notOOF or independent. If all pass
still requires fresh reference/ROI/SAM/Qt before any default-off integration.
No deployment, no newhead, no candidate chosen by oracle or GT.
