# One bounded changed classifier + stricter geometric support trial

All original-pose search finished:289->291/344 but4->6 unmatched rejected.
Two distinct-pixel consistency:289 unchanged, rejected. Original head with
3 detector votes:TRAIN291/4 passed; inner65/0 unchanged, rejected at inner
gain gate, no outer. Exact-pose original+jitter-fold agreement:TRAIN290/6
rejects. Existing production median289/65/33 remains unchanged.

Next one preregistered classifier/pool combination: use existing jitter7262
source-excluded OOF classifier on ALL679 native poses; sameclassargmax/.98,
>=3 distinct actual detector checkpoint votes and maxoneparent/shared5+5.
No original-head .98 AND veto (it prevented any inner newcue); don't relax
newhead .98 or geometry/vote gates. TRAIN currentprefix fixed actual289/4.
Use saved newly extracted native6144 features, verified by original-head
probability replay, and scores from proper held-source fold. GT only after
fixed selection. No image-ID or absoluteXY network input/conditional policy.

Strict all192 source gain/misframesnotincrease/oldloss0/normal0 required.
On failure STOP. On pass fit ONE same400-step original linear head on old
7262 TRAIN feature/label rows, then ALL192 native full-head sourcegate must
pass vs289/4. Then fresh original paired features for already frozen210inner
and209outer native poses, same actual cachedSIFT matrices/source/reference
SHA/context85%. Strict inner>65/80/0,outer>=33/56/1 withloss0/normal0.
First failure stops, no checkpoint fishing or validationfit. Existing detector
weights seeTRAIN: classifier folds only, not detectorOOF/fieldgeneralization.
Even source success requires real reference/ROI/hint/SAM/Qt before release.
