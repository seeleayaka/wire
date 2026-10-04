# Fixed cascade: source-group reverse-pair classifier plus existing geometry rule

Reverse-pair OOF alone297/9 vs installed source295/4 failed. No full head was
fitted. This separate cascade uses exactly the already tested generic median
distinct-voter IoU rank and fixed IoMin.5 novelty filter, with the saved OOF
classifier scores. No fitted geometry, target-driven box edit, new thresholds,
vote count change, selector sweep, additional GT or validation fitting.

All192 source cases and969 candidates; compare against installed source295/4,
not undeployed298 branch. Original predictions, scores and shared5+5 protected.
If OOF TP>295, unmatched<=4, no prior target losses, normal cues0, then fit one
full6144 linear head using exactly the frozen original9808 plus2667 reverse
positive OTHER twins, same400steps seed0 AdamW.01/.001 balanced CE. Full source
must pass the same gate. Classifier OOF only; detectors and prefix are not OOF.

If both gates pass, test the combined cascade on fresh/pinned identical
held-out detections and original-pixel paired features. Reuse a previous fresh
detector run only with exact source/model/config/output pins; re-extract features
for the new head and disclose detector reuse. INNER48 must strictly improve
installed68/80 with0 unmatched and no lost targets; OUTER30 must not regress
40/56+1. Then actual reference/ROI/Qt/SAM gates before any installation.
Failed gate stops without lowering thresholds. All failed artifacts retained.
No E code/deployment, no field or cross-cabinet accuracy claims.
