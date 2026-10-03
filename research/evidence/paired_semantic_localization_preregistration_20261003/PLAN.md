# Train-only geometry hard negatives for paired localization

Standalone paired linear head: source classifier-OOF TRAIN192277->285/344
with4 unmatched unchanged; exact full head292/344 with4. Inner48 full head
63->66/80 but unmatched0->1, rejected. Fresh original reference veto also
fails on inner, same66/80 and1 unmatched. No outer accuracy or deployment.
All four inner additions visualized: three same-classGTIoU>.70; one wire
segment with same-classGTIoU0, annotated connector lower. No GT correction.

Hypothesis: geometry-target supervision is too sparse. Broader visual context
detects a changed cable but can reward a box on the wrong part. Add localization
negatives uniformly for EVERY TRAIN192 GT3/4 target, not failure-image-specific
rules. Use six deterministic boxes: center shifts +/-0.75 GT width along X,
+/-0.75 GT height along Y, centered0.5x and2x dimensions. Do not clip boxes,
change annotations or use heldout GT. Relabel generated training examples
against ALL original TRAIN-source port GT, maximum same-classGTIoU>=.5 ->
that port class, otherwise0. If a shifted box overlaps another GT target,
it remains a positive; avoid contradictory blind-negative labels.

Use the frozen original DINO224 paired source/reference embeddings, existing
1.5x/3x contexts and verified reliable SIFT source->reference matrix. Require
all-context valid reference coverage>=.85; invalid synthetic crops abstain,
not negative. New features6144-D, no coordinates, IDs, IoU or GT at inference.
Keep original1107 samples and all344GT positives; append valid synthetic
training examples only. No inner48 or outer30 feature extraction in fitting.

Controlled first head: SAME linear fit,400 AdamW steps,lr.01/decay.001/class
inverse weights, same source-group3folds, probability.98, precision.98 and
GT-crop recall.25. The single change is enriched geometric training samples.
Score geometry negatives as errors in crop precision but GT-crop recall
denominator stays344, not augmented positives. Only if crop gate passes,
actual original89 proposals ALL192 OOF source gate: TP gain,unmatched<=4,
no old losses,normal0,old5+5. Full-head fit only after this pass,then same
fresh inner48/outer30 source gates. No best checkpoint or threshold sweep.

Reference filtering is a separate already-failed composite, not silently
folded into this trial. If this head fails, keep evidence and diagnose train
geometry versus original proposal failures. All YOLO/body models unchanged;
same-camera repeatedly reused evaluation,not field or cross-cabinet claims.
No auto-deployment; later realROI/reference/Qt/SAM acceptance mandatory.
