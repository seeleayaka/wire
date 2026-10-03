# One TRAIN-only raw-candidate classifier experiment

Accepted native E baseline remains295/344,68/80,40/56;unmatched4/0/1.
Raw-box same-head trial gained2TRAIN but ZEROinner gain, rejected293.47sec.
Freeze its GT-free TRAIN192 proposals(336 after original alignment validity).
Recompute frozen6144 paired vectors for these proposals and normal-reference
self twins, label TRAIN proposals using unchanged weak GT IoU.5 only.
Merge with exact accepted native training corpus7932, source-group deduplicate;
discard conflicting identical crops, never edit GT. All twins stay in original
192 source folds sorted-mod3. Train same linear6144->3 class-balanced CE,
400steps,seed0,AdamW.01/.001, three OOF heads then ONE full head only if source
OOF strict gain over295,unmatched<=4,no old loss,no normal cue. No sweep.
Candidate uses SAME raw poses,three distinct checkpoint votes,p.98,.85 context,
one pose per parent,5+5; current accepted native cues immutable prefix.
Classifier-only OOF; accepted prefix and detectors already saw TRAIN.
No validation labels/images/features fit head. FullTRAIN strict gain also
required. Then independent INNER strict gain/OUTER nonregression, actual
reference/ROI workflow and GUI acceptance before any deployment. Do not
change accepted E head or code in this experiment; failed output preserved.
