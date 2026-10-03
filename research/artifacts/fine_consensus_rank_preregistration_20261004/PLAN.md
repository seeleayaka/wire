# Localization consensus ranking, not confidence maximization

Previous fixed novelty test FINISHED12.14sec: duplicate-only297/5 versus295/4,
six excess duplicate cues removed, one extra unmatched remains. Relative
source-OOF regression on fine boxes added no new eligible coverage (3->3),
so its fresh semantic stage was not run. Preserve all failed evidence.

Hypothesis: selecting the highest semantic probability pose can prefer a poor
box even when another high-confidence pose has stronger detector localization
agreement. Probability of a class is not a localization quality estimate.
No GT at inference, new classifier, threshold reduction, validation fitting,
geometry interpolation, absolute position or image-specific rule.

For ALL969 original fine candidates, compute per-voting-checkpoint best IoU
to actual original/feature/alternative/fine detection rows of the same class,
same source SHA/frame, confidence>.05,16border. Multiple views of a single
weight count ONCE; candidate votes must exactly equal recomputed SHA votes
with bestIoU>=.5. Geometric quality = median(bestIoU over those distinct voters).
Same class argmax/.98/3distinctSHA gate. Within EACH original parent, rank
FIRST by that geometry median, then semantic probability, deterministic box
coordinates/class. Remaining original semantic ordering/budget and .5 IoMin
duplicate exclusion apply to winners; keep old prefix untouched, max5+5.
If a chosen new pair duplicates, exclude/refill finitely with same fixed ranking.
This does not choose a box by labels and does not reuse semantic scores at
changed geometry: no boxes change, all original frozen scores remain valid.

ALL192 source netTP>295/unmatched<=4/no old loss/zero normal cues required.
Failure stops without INNER/OUTER. If passed, fresh fine detections, original
pixels/geometry scores and exact SAME selector on INNER (strictTP>68/0errors)
and OUTER(non-regression40/1), then actual reference/ROI/Qt/SAM validation.
Independent all-source geometry/ranking/GT replay before any adoption.
One median-IoU ranking, no minimum/maximum/weight/threshold sweeps.
Development scenes repeatedly used, not independent field/generalization proof.
