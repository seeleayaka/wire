# Unanimous paired committee with mandatory ORIGINAL reference veto

Standalone committee passes train285/344 with4 unmatched and inner66/80
with0, but fails outer: an added candidate increases unmatched. Preserve the
standalone failure and its saved predictions. No two-member voting change,
probability change, label correction or image-specific rule.

Use the SAME frozen all3-member>=.98 committee candidates at the same native
boxes and shared5+5, followed by original normal-reference veto (same class,
reference>.25,alignedIoU>=.5). Fresh teacher and first-student full normal073
and640/960 matched reference views for EVERY candidate source on ALL270.
Exactly the reference rule already implemented and tested, no new thresholds.
Do not backfill after veto. Cached source committee/fresh reference is an
explicit source-only preflight, not actual currentROI/Qt/SAM acceptance.

TRAIN192 then inner48 require net gain/no unmatched increase/no old loss/
normal0; outer30 no TP regression/unmatched<=1/no old loss/normal0. Stop after
any stage failure. All source/head/normal-reference hashes protected. Three
source-fold heads are correlated and TRAIN committee is not OOF accuracy.
Outer reuse influenced composing an existing mandatory veto; this is not an
untouched independent test. Never assert field or cross-cabinet performance.
No auto-deployment. A pass still needs fresh original full workflow/ROIs,
fresh source proposal parity and actualQt/SAM validation.
