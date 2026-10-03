# Frozen accepted head, median proposal actual source test

TRAIN192 median candidate ceiling301 equals old extended ceiling301; one
missed TRAIN target has newly adequate median IoU but another is lost.
Ceiling alone is not gain; fixed current head can behave differently with
new crops. Evaluate actual SAME accepted238f517b paired head, probability.98,
central4x4/CLS6144. No retraining, new threshold or GT-driven candidate choice.
Fresh every missing exact1.5/3 crop signature, original reference homography,
valid coverage.85; never reuse expected/expected vectors as inspection pairs.
Same acceptedFULL286 old prefix, original select/shared5+5. Stop if no TRAIN
gain or extra unmatched/lost/normal cue. If TRAIN passes, construct GT-free
median candidates inner48 then outer30 from same already-existing model
outputs and run fresh paired features. Fixed inner gain64, outernonregress33;
unmatched0/1 not increased. A source pass still requires actual reference/ROI/
SAM/Qt before default-off integration. No changes to independent jitter run.
