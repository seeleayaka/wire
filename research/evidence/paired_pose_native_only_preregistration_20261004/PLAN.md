# Frozen new classifier as replacement ONLY for newly searched proposals

Pre-run evidence: native-trained auxiliary sourceOOF290/4 and full291/4
pass; inner65/0 unchanged. ALL210 fresh inner native proposals have ZERO
old-head same-class p>=.98. Consequently requiring the OLD head as well as
the new one mathematically prevents any inner gain, regardless of improved
native classifier training. Do not lower old thresholds to fit examples.

Hypothesis: use the already trained source-grouped native classifier itself
for new pose proposals, while keeping the entire accepted median prefix
and its original classifier untouched. No retraining or probability tuning.
Same>=2 unique checkpoint votes, native geometry, context85%, p.98,
one alert per parent and shared5+5. Use sourceOOF heads for ALL192 source gate;
if it passes, use the same frozen full head for ALL192 full-model recheck.
Only then use the exact previously fresh inner48 features/scores, after
independent complete SHA/selection audit; only then fresh outer30 inference.

Each gate: strict train and inner TP increase, outer nonregression, unmatched
not increased, no lost old matches, normal zero. First failure stops, no E
deployment. Save immutable selection/score/pin evidence. Reused validation
is development evidence, not a new independent generalization/field test.
Actual reference/ROI/SAM/Qt still required after any source pass.
