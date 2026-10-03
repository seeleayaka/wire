# Fixed paired classifier plus ORIGINAL reference veto

The standalone paired classifier remains REJECTED on inner48:63->66/80,
unmatched0->1. Its classifier-OOF training gate was277->285/344, unmatched4
unchanged. These are reused same-camera data, not field performance.

New composite hypothesis, before any new reference prediction: unchanged
reference connector detections may veto geometry that a paired embedding
classifier confuses with unplugged state. Keep the trained full linear head,
probability.98, same two detector proposals>.05, same class agreement, source
GT IoU.5 and existing5+5 budget unchanged. No image-specific restrictions,
new threshold, GT edit, best head selection or re-training.

First score the EXACT full head on TRAIN192 frozen source/reference features;
OOF head results are not substituted for the checkpoint intended for runtime.
Generate selections before GT scoring. Preserve every currentV3 native cue.

For each selected new native candidate, use verified source->reference
homography and original aligned_predictions. Fresh normal073 full reference
predictions from BOTH fixed teacher and first student checkpoints, plus their
original640/960 matched seed views. Original reference veto: same class,
reference confidence>.25 and aligned IoU>=.5. This can only remove new native
candidates, never adjust their box, replace old cues or turn reference
non-detection into a physical fault verdict. Apply AFTER semantic candidate
budget; do not backfill rejected candidates to change the experiment.

Layered TRAIN192 then inner48 source gates: net TP gain, unmatched no higher
than old baseline, no old losses and normal0. Only if both pass, run fresh
paired DINO/reference registration and the same veto on outer30; outer must
not regress or add unmatched. Preserve all failures and truthfully state
cached source inference vs fresh reference inference.

This is a source-only diagnostic composite, not actual current ROI/Qt/SAM
acceptance. Even a pass must then be checked with fresh whole-image workflow,
unchanged recorded ROI gates, source proposal parity, actual GUI and SAM before
optional default-off integration. No automatic deployment. Reused inner
influenced composing with existing reference gates; no independent evaluation
or cross-cabinet accuracy claim.
