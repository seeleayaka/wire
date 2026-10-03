# Unique cross-detector evidence components, no NMS threshold change

Standalone committee train285/344+4 unmatched,inner66/80+0,outer38/56+2,
outer rejected against old33/56+1. Fresh existing reference veto leaves
these results unchanged. Keep failures; do not relax the unmatched gate.

Hypothesis: two native candidate boxes can both cite the SAME opposite-model
detection, despite mutualIoU<.5. Counting checkpoint SHAs without detection
identity admits duplicated evidence for one object. Check this uniformly over
ALL270 sources, not a filename exception. No GT required to build the graph.

Build bipartite graph from the EXACT existing teacher/first-student merged
native detections at original>.05 floor and original complete-frame margin.
Edges only distinct checkpoint heads,same class,IoU>=.5. Stable node IDs are
(checkpoint SHA,merged-detection index), not source coordinates as features.
Each selected committee proposal is linked exactly to its original detector
node using original detector-score provenance and native geometry. An
unlinked/ambiguous node is explicit experimental failure, not a false negative.

Among NEW committee additions in the same connected evidence component, keep
only the original highest minimum-member score, same existing tie order. No
old cue removed, no native box changed, no threshold lowered and no backfill.
Components represent shared evidence, not confirmed physical-object identity;
nearby true objects may share ambiguous support and be conservatively limited.

ALL192/inner48 require net gain/no unmatched increase/no old loss/normal0;
outer30 no regression/unmatched<=1. Score after filtered outputs saved. Full
source detector and committee caches hash reverified; this is a cached source
policy preflight only. Passing still requires fresh source proposal/component
parity, original reference/ROI gates and actualQt/SAM. Three classifiers and
YOLO votes correlated, training committee notOOF, reused validation influenced
design, no independent field/cross-cabinet claim. No automatic deployment.
