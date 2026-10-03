# Uniform median-seed local pose search, same accepted classifier

Motivation from accepted median geometry289/65/33 vs286/64/33: input geometry
changes existing frozen classifier acceptance. Rectangle4166 cropP93.60%
rejects; its predeclared3134 controlP100%,GT225/344 passes but actual source
289->289, no gain. Positive-jitter original context likewise sourcezero.
Do not retrain or change thresholds; no photo/position-specific patches.

All median weak seeds that do not IoU.5-overlap stronger fixed289/65/33 old
prefix receive SAME6 local poses already defined by uniform training jitter:
center four directions15% of width/height, centered size0.8 and1.2, plus seed
identity. Each actual proposed box rechecks score>.05 full16pixel boundary
and SAME-class IoU.5 support by>=2 distinct existing checkpoint SHAs; views
of same weight count once. If no slots, short circuit. No GT at generation.
Original square1.5/3 contexts with valid coverage85%, same frozen DINO1536
paired6144/head238f517b and probability.98. Retain raw probabilities.

One alert maximum per explicit parent seed, regardless scale pairIoU. Choose
highest raw eligible same-class probability>=.98 within parent, deterministic
box tie ordering, then original select handles old-prefix/globalIoU/shared5+5.
This is proposal search, not calibrated multi-test probability or physical
fault certainty. Head scores are not amplified or renormalized.

TRAIN192 first against stronger median289/344/4: strict new matched target
gain, unmatched not higher, old matches preserved and normalzero. Stop first
failed source gate; if passes fresh inner48 against65/80/0 then outer30 against
33/56/1. All candidate features and selections emitted before unchangedGT
scoring. Actual reference/ROI/warp/SAM/Qt needed before release. Existing
accepted median SAM job/core/head/config untouched. No deployment, purchases,
quota resets, downloads or subagents. Same reused data, not field validation.
