# One fine-candidate TRAIN-only hard-negative head

The frozen native head on fine detections gained2TRAIN but added7unmatched
boxes(297/11 vs295/4), rejected361.01sec. Do not deploy or relax gates.
Freeze all969 GT-free fine-native candidates from completeTRAIN192 inference.
Add their actual paired vectors and969normal-reference-self twins to the
EXACT accepted7932 TRAIN corpus, source-group deduplicate/drop conflicting
identical crops. Same192mod3 source folds; whole-source twins excluded together.
Same6144->3 linear head,400steps seed0 classbalancedCE/AdamW.01/.001.
No hyperparameter sweep/no new detector/no validation fitting. OOFincremental
classifier source strict net gain/no extra unmatched versus295/4 before ONE
fullhead. Preserve accepted cues and3distinctcheckpoint/.98/.85/5+5.
If source/fullTRAIN gates pass, fresh fine INNER48 requires strict gain and
no unmatched, then OUTER30nonregression; real reference/ROI/GUI gates remain
mandatory before installing anything. Detector/prefix notOOF; same-scene
repeated development, not field accuracy. All unsuccessful artifacts retained.
