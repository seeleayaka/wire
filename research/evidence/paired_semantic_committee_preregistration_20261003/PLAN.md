# Fixed source-fold unanimity, no threshold sweep

Standalone full paired head inner63->66/80 and unmatched0->1 is rejected;
fresh original reference veto did not improve it. Uniform TRAIN localization
negative preparation is running independently. This cheap separate hypothesis
uses the THREE EXISTING fixed source-fold heads, not retraining, model search
or selecting best members. Require ALL three members to predict exactly the
two-detector proposed port class with individual probability>=.98. Keep the
same3-class outputs, contexts, native geometry, existingtwo-checkpoint>.05
proposals, original shared5+5 and all currentV3 cues unchanged. No averaging
to hide a weak vote, score relaxation, image-specific exclusions or GT edits.

Effective review score is minimum of member probabilities, NOT a calibrated
fault confidence or new single softmax. Persist every raw3-class distribution
and all3 checkpoint SHAs. The three classifier heads share training sources
and the underlying YOLO proposals, so they are correlated, not independent
physical evidence. TRAIN source committee performance is not OOF: two members
have seen each source; one is held out. Do not relabel it OOF accuracy.

TRAIN192 first, net TP gain/no unmatched increase/no old loss/normal0; then
fresh inner48 and outer30 paired crop inference with all-member criterion.
Inner requires gain and unmatched0; outer no regression and unmatched<=1.
No choosing2-of3 or another cutoff after observing results. No reference-veto
composition silently folded in. Same-camera reused validation, no field or
cross-cabinet accuracy. A pass still requires real currentROI/source parity,
Qt/SAM acceptance before any optional default-off integration. No auto-deploy.
