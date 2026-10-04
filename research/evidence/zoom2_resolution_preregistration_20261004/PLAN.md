# Fixed 2x physical-resolution control of evidence acquisition

The ALL192 evidence-first trial failed:298/344+4 ->298/344+6, normal1,
157actions in89sources,2002finalproposals; independent replay passed.
ALL46 prefix misses had only3 exact-two-voter seeds,1 selected seed and0
final covering proposals. Those are NON-NESTED posthoc coverage sets, not
proof that semantic scoring is the bottleneck. Additional read-only analysis
shows30 misses in budget-full images; these cannot be recovered by additive
selection under the protected prefix/5+5 contract. Of16 other misses, raw
IoU.5 distinct-checkpoint coverage is0votes:10,1vote:3,2votes:3.

The proposed footprint pooling is already a failed20261003 experiment and
must not be rerun or represented as new. The next bounded hypothesis is
upstream sampling: frozen detector checkpoints may fail small components
at native960-crop/960-input sampling. Test exactly one physical view scale:
480x480 source crop resized internally to960, two offsets(-60,-60)/(60,60).
No image/GT-specific sizing or scale sweep. Same original157actions selected
by the fixed geometry-only policy, same one missing checkpoint for each.
This is an isolated SCALE control, not a new weight, new vote or guarantee
of gain. It cannot solve the budget-full misses and may sacrifice context.

Reuse prepared ALL192 train cases and exact original298/4 research prefix,
not failed ROI outputs. Require identical action lists to the rejected
960 experiment. Preserve source coordinates returned by YOLO (already
rescaled to original480 crop), then add window origin ONCE; never divide
boxes by2 again. Artificial-edge rejection remains16 SOURCE pixels. All
downstream proposal poses, original-pixel DINO features and9459 head remain
unchanged; final p.98, three distinct SHA votes, coverage.85, IoMin.5 and5+5.
Same checkpoint at two views remains one vote. No training or GT inference.

Run geometry/range/scale mapping tests and import tests before ALL192.
Single worker,2CPUthreads, no SAM, source deadline7200sec. On failure save
progress and leave mainline unchanged. Do not modify E or active inputs.
Pin old rejected report, successful audit, prepared cases, helpers and models.
Independently replay action identity,480 geometry, source-coordinate bounds,
voter uniqueness, cached-feature probabilities, rank and ALL192 GT metrics.

Source gate:TP>298,unmatched<=4,no old target lost,normal0. If it fails,
do not run validation or select a more favourable threshold. If it passes,
fresh combined-cascade INNER48 must exceed68/80 with0unmatched, then OUTER30
at least40/56 with<=1unmatched; actual reference/ROI/Qt/SAM acceptance before
any deployment. These repeatedly used development sets are not field or
cross-cabinet accuracy. No paid compute, no new model downloads.
