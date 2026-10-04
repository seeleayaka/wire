# Resolve missing checkpoint evidence with bounded source-only local views

Motivation is TRAIN49 miss taxonomy (19 have partial detector coverage without
three checkpoint agreement), not tuning against named validation images.
The prior fine median-rank SOURCE298/4 has passed source replay but validation
is still running; it is not installed. This is a separate combined research
branch, preserving that research prefix and all accepted295 cues, not a claim
that298 is the mainline. No final cue can be accepted with only two weights.

Preparation: ALL192 TRAIN, original four source model outputs plus actual
teacher/student fine views. Four streams are ONLY THREE distinct weight SHAs:
alternative is another feature-checkpoint view, not a fourth independent vote.
Generate same seven uniform poses,16border/.05floor/IoU.5, duplicate IoMin.5,
original registration and .85 context, remaining shared5+5 budget. Retain only
EXACTLY TWO distinct voter-SHAs as unresolved SEEDS, not output cues. No GT
used to select seeds. Every source/model/pixel/alignment/plan/helper is pinned.

Only after the previous held-out worker ends and >=6GiB RAM, score all prepared
two-voter seeds with the accepted9459 head and ORIGINAL paired DINO pixels.
Semantic argmax matching class andp.98 required. Per parent rank median best
voter-IoU then semantic-p/coordinates/class; choose up to remaining budget
nonduplicate seeds. Invoke ONLY the missing checkpoint using two shifted local
960x960 views centered with offsets±120 (same fraction as earlier1280±160),
input960,conf.001/iou.7/max_det300,16 artificial-edge exclusion. Views of that
same checkpoint remain ONE vote. No model training, label fitting, threshold
sweep, normal-reference replacement, photometric pixel changes or SAM.

New ROI predictions map back to original coordinates. Merge as actual extra
model evidence, same .05floor/3distinct-SHA/IoU.5/16border/.85 gates, recompute
unchanged7poses from actual boxes and ORIGINAL paired DINO scores at final
geometry, median voter-IoU rank and IoMin.5 duplicate rule. Protect all298
research-prefix source matches and original295 current fields, shared5+5.
Require SOURCE TP>298,unmatched<=4,no lost targets,zero normal cues before
fresh validation of the COMBINED cascade (baseline currentINNER68/0,OUTER40/1),
and actual reference/ROI/Qt/SAM acceptance. Failed source stops without holds.

Supplemental source preparation readiness reports input coverage/seed counts
ONLY, not accuracy. No deployment or cross-cabinet/field accuracy claims.
Do not edit any existing live worker code or mainline fingerprints.
