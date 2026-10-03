# Upstream candidate coverage: one finer tile scale

Accepted median289/65/33 remains unchanged. All subsequent pose, consistency,
classifier agreement, threevote and existing-boundary controls rejected.
TRAIN pose remaining53 misses:20 fullsharedbudget,17 no valid posegeometry,
16 semantic belowgate. Head-only refinement cannot create absent boxes.
Existing teacher1280-input A/B already ran (inner not accepted), so do NOT
repeat that resolution change. Feature1280 views already used in baseline.

One physical tile scale change: source windows960x960 stride720 vs original
1280x1280 stride960, keeping original predict_imgsz960,all floors/.7 NMS/
max_det300/edge16/cross-tile.5. Both original teacher AND student infer at
new scale; same checkpoint views count ONCE in geometric support. Sameclass
median geometry supports>=2 distinct checkpoint SHAs including original
feature/feature1280 caches. Remove overlaps with frozen stronger current
prefix, fixedaccepted238 pairedhead/contexts1.5/3/.98,old5+5 unchanged.

GT-free full eligibility: remaining extra slot AND any original detector weak
row>.05. Smoke first2 eligible TRAIN sources with any existing selectedcue,
sorted by source list, never GT. Smoke checks outputs/latency/shape/hash/config
restoration only, not accuracy. Full ALL192 source first: strictgain over289,
unmatched<=4,no oldloss,normal0; success only thenfreshinner>65/80/0 and
outer>=33/56/1. First failure stops. No labels for window selection. No retrain,
downloads/quota resets/purchase/newagents; no E runtime/UI changes. Original
reference/ROI/live/nativehint/SAM/Qt still required before any release.
