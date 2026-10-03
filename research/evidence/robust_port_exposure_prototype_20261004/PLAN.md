# Photometric compensation prototype, not deployed

Prepare a general fixed estimator while ALL78 exposure audit runs. Registered
expected/observed log-channel ratios, median on uniform32 grid, valid coverage
>.99, unsaturated16..239,min512samples/channel, gain.7..1.4,MAD<=.12,
near-identity2.5% skip. No source names, absolute box coordinates, GT or labels.
Never replace actual DINO/visual-report image pixels. Candidate detector-only
branch may use a compensated copy in a later separately pinned experiment.
No new source fitting or threshold sweep; photometric invariance is a
prototype hypothesis, not recognition accuracy. Significant structural changes
remain in source; uncertain registration/shading/insufficient samples retain
original pixels. Synthetic unit tests cover multipliers.8/.85/1.15 and colour
channels, changes, identities and fail-closed cases. Must independently test
recognition and false alarms across full original/perturbed cohorts before E.
