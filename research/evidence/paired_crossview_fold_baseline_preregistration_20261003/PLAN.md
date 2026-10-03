# Like-for-like classifier fold diagnostic, not deployment acceptance

Crossview/context-footprint cropOOF passesP98.1132/R59.88; original and
expanded candidate experiments both rejected deployment-prefix sourceOOF
zero additions after geometry FULL286. Keep those rejections. Examine a
different diagnostic: same source-held classifier-fold baseline versus
fold-baseline plus crossview. Teacher detectors still trained ALL192, so
this is NOT full detector OOF or field accuracy. Do not call281 the deployed
baseline or claim this diagnostic itself improves the deployed system.

Current sourcefold geometry is281/344/4, sourcefull geometry286/344/4.
Exact saved source-native context/footprint OOF scores on original89 weak
proposals are reused; no head retraining, image cues, threshold tuning or GT
selection. Keep same OOF geometry predictions as prefix and shared5+5.
Report both own before/after counts. Diagnostic must gainTP/no error/oldloss/
normal0. If it passes, a NEW final full-head composition must still meet
actual deployment-prefix TRAIN286gain/no error,inner64gain/noerror/outer33
nonregression plus realreference/ROI/SAM/Qt before any integration. No gate
relaxation, no automatic deployment or rewriting prior result histories.
