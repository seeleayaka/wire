# Reverse positive paired TRAIN examples to explicit OTHER direction controls

Hypothesis: current linear paired classifier can conflate bidirectional visual
difference with a fault present in the observed crop. Positive forward pairs
have faulty observed crops and the aligned normal073 reference as expected.
Reverse those pairs into normal-reference observed / faulty expected, labelled
OTHER. This matches the system's observed-fault semantics, not an unchanged
scene claim. The original positive and negative examples remain intact.

One training variable only: append reverse twins of all positive examples in
the audited9808 TRAIN corpus. No new detector, pixels, descriptor, feature
normalization, architecture, weighting rule, selector or probability threshold.
Swap first two1536 feature blocks; absolute difference and product invariant.
All twins keep the original source fold.192 TRAINmodulo3 folds, fixed6144->3
linear head,400steps seed0 balanced crossentropy AdamW.01/.001. Class weights
are recomputed by the unchanged fit function on each augmented training fold.

Frozen969 fine-native proposals and features remain the evaluation population.
Installed295/344+4 unmatched source prefix preserved. Same p.98,3 distinct
checkpoint SHAs,.85 context, geometric rules and5+5; original fine selector,
not the later median-IoU selector, isolates the effect of training augmentation.
Source-group classifier OOF does NOT make the detector/prefix OOF. No inference
feature includes filenames, absolute coordinates, target associations or GT.
TRAIN strict gain,unmatched<=4,no original losses,zero normal cues before one
full head. If either source gate fails, stop before validation and deployment.
If both pass, fresh INNER48 must improve68/80 without unmatched; OUTER30 must
not regress40/56+1, then actual reference/ROI/Qt/SAM acceptance is mandatory.

No GT modification, no reference replacement, no holdout fitting, no threshold
sweep, no installed-mainline changes. Repeated same-scene development results
are not cross-cabinet/field accuracy. Keep all failed artifacts.
