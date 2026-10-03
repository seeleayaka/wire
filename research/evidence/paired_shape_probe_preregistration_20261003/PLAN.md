# Shape identifiability audit and finite train-only probe

Hypotheses before probes: (1) original paired crops omit the shorter candidate
axis; same center/max side makes identical input despite wrong boxIoU;
(2) original low-confidence detector candidates are mislocalized even if
objects exist; (3) reference warp causes mismatched crops. Test1 with crop
signatures and real tensors at both scales before any training. This is a
representation audit, not a field accuracy claim or relaxation of GT.

If confirmed, append three normalized candidate shape descriptors w/maxside,
h/maxside,area/maxside^2 to original paired6144 features. No absolute x/y,
image name, GT metadata, source id or scene-specific thresholds at inference.
Shorter-axis shrink0.25/0.5 and expansion to the unchanged longer-axis extent
applies uniformly to all originalTRAIN344GT; label against all actual original
same-source GTIoU0.5. Reuse a GT vector only if both crop floor/ceil signatures
are exactly identical, otherwise abstain. Keep all existing3134 training rows.

Fixed original sourcefold3,400 AdamW.01/.001 steps,class balancing,seed0,
probability.98,cropP.98/GTrecall.25. If fails, reject without source inference.
Then use actual novel weak candidate TRAIN embeddings to evaluate OOF head
as an additive extension AFTER the fixed geometry head286/344/4 baseline,
preserving old hints and remaining5+5. Must increaseTP without unmatched or
old target loss, normal0. No heavy heldout rerun if train has no net gain.
Only after success fit fixed full head and prepare fresh48inner/30outer
embeddings, composite baseline64/80/0 and33/56/1, require innergain/outer
nonregression. Existing validation has been used before, not independent.

Do not modify the release or complete-SAM pinned code/weights/manifest.
No deployment, callbacks or newSAM jobs from this probe. Single CPU thread;
no DINO encoder forward while existing SAM job occupies memory.
