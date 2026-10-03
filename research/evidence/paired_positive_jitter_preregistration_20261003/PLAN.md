# Uniform near-correct localization tolerance, original frozen descriptor

Representation/context-footprint experiments remain rejected for zero source
gain or crop gate failures. The like-for-like fold check also zero gain:
281->281,4 errors, not a deployment result. No gate lowering or reversal.
Current geometry TRAIN3134[2756,184,194] versus base1107[739,184,184] means
2027 synthetic examples consist2017negative/10incidental neighboring-target
positive. Fixed shifts.75width/height and scales.5/2 target hard wrong boxes,
not systematic near-correct detector-box tolerance. This motivates a different
train-only augmentation, not new architecture or metadata feature channels.

For every originalTRAIN344GT apply uniform center shifts±.15width/height in
four cardinal directions plus centered .8/1.2 scales. These generally remain
GTIoU>=.5; always relabel against ALL original same-source GT, not blindly
positive. Context coverage.85/originalSIFT reliability gate unchanged; invalid
examples abstain, no label edit. For every valid new observed/expected sample
add expected/expected normal-reference-self negative at same crop geometry.
Reuse original3134 accepted geometry features unchanged; new crops fresh
frozenDINO1.5/3CLS+central4x4,6144paired, no shape/id/XY features. No backbone
training. Sourcefolds192 sortedmod3 unchanged, no heldout supervision.

Smoke first2eligible TRAIN sources then finite full192 feature preparation;
2CPUthreads. Fixed original linear400AdamW.01/.001/class balancing/seed0,
OOF3/probability.98/P.98/GTrecall.25. Pass requires new OOF actualTRAIN gain
after acceptedFULL286/344/4 prefix/no unmatched/no oldloss/normal0, not weaker
281/277 baseline. Then fixed full head exactTRAIN gain, freshinner48 vs64/80/0
gain/no errors andouter30 vs33/56/1 nonregression. Failure stops before next
stage. Positive augmentation itself is not accuracy or weak-head acceptance.
No automatic integration; success still needs newreference/ROI/SAM/Qt.
Preserve existing default-off source286/64/33/SAM complete policy and all old
models/GT/calibration. No filename-specific shifts, score/epoch search or field
accuracy claim. All prior validation reused, detectors trainedALL192.
