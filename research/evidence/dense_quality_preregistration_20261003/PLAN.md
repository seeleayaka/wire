# Joint localization-quality supervision, 2026-10-03

Plain24 crop P83.44%/R57.01% failed90% gate; lower training loss was not a
reliable accuracy gain. Next change the training objective, not output scores.
Fixed original fine epoch8 checkpoint plus16 additional plain epochs,24 total,
same architecture/data/optimizer/order and last-only choice. Freeze DINO.

For each existing labeled center, detach the actual IoU of its predicted box
against the original GT rectangle and use that continuous value as the class
confidence target. Quality focal BCE weighted by |p-target|^2; background target
zero, original Gaussian negative suppression. Keep smoothL1 regression and add
GIoU on the same GT centers, making overlap directly part of training quality.
GT used in TRAINING ONLY. Decoder accepts only predicted logits/boxes/shapes,
no target access. Same score0.5/P>=90%/R>=25% feasibility and same source
candidate0.75/two-tile0.5/IoU0.5/Current V3/shared5+5 gates. No score sweep,
label edits, class filter or coordinate-specific rules. No best epoch selection.

Inspired by continuous localization-quality supervision in Generalized Focal
Loss (https://arxiv.org/abs/2006.04388), not a full GFL architecture or a claim
of reproducing its results: no distributed-box head and our existing center
decoder/weak rectangle labels remain. Only local source gates can qualify it.
Reused validation is exploratory, not independent accuracy. No automatic
deployment; fresh source-reference/GUI/SAM acceptance remains mandatory.
