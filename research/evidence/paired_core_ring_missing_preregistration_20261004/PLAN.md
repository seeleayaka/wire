# Missing-evidence-safe revision, declared before any classifier fitting

The first body/ring trial stopped184.58sec before optimizer/head/evaluation:
an accepted square DINO context is not proof of .85 exact-body/ring coverage.
Keep its code, plan and failed output unchanged. A separate ALL192 geometry
readiness audit examines9808+969 examples without labels or success scoring.

This revision retains ALL9808 training examples, no hard-example filtering or
threshold loosening. Same46 fixed GT-free body/ring descriptor statistics;
append one evidence-present flag, dimensions47. When coverage is insufficient
or a quadrant/gradient has no valid support, use47zeros as explicitly missing
input. Invalid image shape, nonfinite values or invalid boxes still raise.
All valid descriptors have presence1. Raw candidates missing body/ring evidence
must be forced to probabilities[1,0,0] BEFORE the existing append selector;
therefore they can never generate a new cue. Old accepted cues stay unchanged.

Reference-self twin pixel provenance/folds retained. No image IDs, absolute
coordinates, labels, compensation gains/status as descriptor inputs. Sole head
is6191->3 linear with seed0/400/AdamW.01/.001/class-balanced loss; append47
dimensions standardized using TRAIN-fold mean/std floor.01, clamp[-5,5].
Same source-modulo3 exclusions, .98/3distinctSHA/.85/5+5 protected prefix.
Same strict295TP/4unmatched TRAIN OOF net gain gate, then full head, separate
fresh INNER strict gain/OUTER non-regression/actual reference-ROI/Qt-SAM gates.
Reject after failed OOF; no full head, validation access or E deployment.

Original failed trial pins remain intact; new driver/adapter/plan/output pins.
Re-extract all pixel descriptors, independently recompute train-only moments,
scores (including compulsory missing-input override), selectors and GT counts.
Ten original descriptor tests plus new absent-data/invalid-input/abstention/
train-only normalization tests must pass before fitting. This is a finite
representation trial, not a capacity/threshold sweep or field-accuracy claim.
