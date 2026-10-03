# Native proposal hard-negative learning, prospective protocol

Diagnosis: native max-pose search adds two TRAIN matches and two unmatched
boxes. Actual unchanged reference/ROI/warp filters do not fix the error.
GT-jitter-only classifier agreement also failed. Neither branch is deployed.

Ranked hypothesis: a classifier trained on actual searched proposals and their
background errors, rather than only GT-near shifts, can reject confusing
native crops while retaining new correct detections. Alternate hypothesis:
finer upstream physical tiles can recover genuinely missing proposals;
that independent finite job remains running and is not modified.

Training only: existing7262 paired6144 features plus all679 real native pose
features. Native labels use unchanged TRAIN GT same-class IoU>=.5 only after
the GT-free crops/features were already frozen. Entire image source folds
remain sorted TRAIN names modulo3. All augmentation/reference twins stay in
the source group; YOLO is NOT source-independent and results are NOT field
accuracy. No source names/coordinates/labels enter classifier features.

Deduplicate exact two-scale pixel crop signatures within each source and
observed/expected-self mode. Drop every member of contradictory-label
signature groups. Record duplicates/conflicts and verify repeated vectors.
Fixed existing linear6144->3, seed0,400 AdamW steps, class-balanced CE,
lr.01/weight_decay.001; no sweeps, selected epochs or calibrated thresholds.
SourceOOF original and new classifier must BOTH accept SAME native pose at
unchanged p.98. Keep original scores, >=2 distinct checkpoint SHA support,
one parent seed alert, all original boxes, shared5+5 and context85%.

First source gate: all192 versus289/344/4, strict TP gain, no increase in
unmatched boxes, no lost matches and zero normal cues. If it fails, stop.
If it passes, fit exactly one full head and repeat all192 with that head;
same source gate. Then fresh native inner48 must improve65/80/0, outer30
must retain>=33/56/1, no old losses/normal cues. First failure stops, no
automatic deployment. Save complete predictions, SHA pins, feature/label
metadata and model snapshots. Later actual reference/ROI/SAM/Qt acceptance
and independent audit are still required for any adopted change.

These reused cohorts are repeatedly examined development data, NOT a new
independent generalization test or proof of cross-cabinet reliability.
