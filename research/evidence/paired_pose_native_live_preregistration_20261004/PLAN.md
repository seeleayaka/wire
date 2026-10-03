# Actual final filters for every new native-trained pose source

Source gates and independent replay pass: classifierOOF291/4, fullTRAIN295/4,
inner68/0, outer40/1 versus accepted289/4,65/0,33/1. These are reused
development cohorts and weak GT boxes, not physical continuity/field accuracy.

Run every prospective new-cue source (6 TRAIN,2 inner,4 outer), all four old
median-gain sources, both rejected pose false-cue sources, plus first normal
per cohort, deduplicated source/stage. New actual InitialReviewWorker SIFT/
DINO, actual accepted median backend, new native head only for new proposals,
>=3 checkpoint SHA, unchanged normal-reference full+640/960, ROI/.98warp,
duplicate/shared5+5, exact forward-mapped native-hint linkage. Keep every
original field except supplementary append exactly unchanged.

New head9459ba29..., encoder/input6144/known classes validated; bind the real
median runtime fingerprint and head/helper/source/audit SHA before and after.
Exact raw proposal class/score/geometry parity with frozen source expected.
Failures are recorded, never quietly counted as valid source results.
Normal local-ECC fallback may abstain with zero cues; record separately and
never call it successful recognition. SAM deliberately pending until final
source and safety pass. No E install or automatic verdict from this trial.

Recount ALL270 using actual accepted native additions for every prospective
source, baseline for no-addition sources. Strict TRAIN/inner gain, outer
nonregression, no extra unmatched or lost matches/normal cues. Only then
stage default-off integration and fresh SAM/Agent/actual Qt acceptance.
