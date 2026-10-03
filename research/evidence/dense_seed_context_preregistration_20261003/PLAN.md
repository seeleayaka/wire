# General seed-region relocalization, 2026-10-03

Hypotheses, ranked: (1) fixed YOLO boxes lack useful geometry and an independent
RGB/DINO center/box decoder can relocalize nearby objects; (2) small native
contexts retain pixels lost when1280 crops are resized448; (3) some apparent
falseobjects are normal real connectors and would need reference novelty checks;
(4) strict support/budget can still limit recall. The exact symmetric source
candidate oracle proved no inner gain, so its slow paired experiment was NOT run.

The previous standalone epoch8 fine head stays rejected at crop P84%. This is
a new composite algorithm, NOT a claim that its crop gate passed. Use that
fixed checkpoint as a relocalizer inside weak proposals from BOTH always-real
teacher and first-student raw predictions. No partially inferred feature-peer
cache participates. Head output must satisfy both strict context scores0.75,
same-class IoU0.5 agreement across640 and960 native crops, and at least one
same-class seed support with score>0.25 and IoU0.5. No score threshold search.

Seed selection: valid complete frame-margin16 rows over0.25, not overlapping
any current accepted box at class-agnostic IoU0.5. Sort confidence, geometry,
class; deduplicate same-class seeds at IoU0.5; top6 maximum like current recheck
budget. Two fixed centered square contexts640/960, clamp within frame; no
GT coordinates, filename cases, detector-specific target exceptions. Reject
crop artificial-edge detections16 pixels; preserve current primary and all
extra cues; all extra combined max5. Failed encoder/head/input/SHA must fail
explicitly, never pretend that a skipped inference was a negative observation.

ALL192 training sources first, then48 inner,30 outer; source score label class
IoU0.5 unchanged. Need net train AND inner gain, outer nonregression, no new
unmatched/old target loss/normal cues. This source gate excludes deployment;
fresh reference/ROI/registration and GUI/SAM tests mandatory after any pass.
All thresholds fixed, last epoch8 checkpoint fixed, no more training/selecting
on reused inner data. Reused benchmark, not independent field accuracy.
