# Refine existing geometry rather than append more boxes

All pose variants/consistency/agreement/threevote controls rejected, accepted
median289/65/33 unchanged. Hypotheses:1 old selected box bounds are inaccurate
but nearby3-model geometry agrees;2 topcontext classifier scores do not improve
localization;3 budget/full lack-of-proposal misses remain upstream constraints.

One parameter-free coordinate median proposal for EACH existing selectedbox
from SAMEclass IoU.5 weakscore>.05 complete16px witnesses. One highestscore
box per distinct checkpoint; require>=3 SHAs before AND after median. Same
object originalboxIoU.5; reject neighborIoU.5 collision; skip <1px changes.
Only refine if fixedaccepted238 head sameclassargmax/.98 with original1.5/3
context85%/6144 features. Retain olddetectorconfidence, raw partition arrays
and sourcecounts. No image names/absolute position rules/GT runtime input.
New all_predictions is a separate geometry trial NOT existing GUI hintpayload.

ALL192 TRAIN first vs289/344/4, strictTPgain OR unmatchedreduction plus no
oldmatchedloss/normal0. Source selection saved BEFORE originalGT scoring.
If fail STOP before validation. If passes freshinner48/outer30 require no
oldloss/unmatchedincrease and nonregression in both groups. Protocol clarified
BEFORE execution: inner already has65 boxes/65 matches/0unmatched; a fixed-
cardinality geometry-only trial cannot exceed65 matches. Requiring an inner
newmatch without adding boxes would be an impossible gate, not independence.
Same head/runtime/models/labels/oldcue records/E pipeline protected. AllGT
weakbox labels and reuseddata; no field/crosscabinet or physicalfault claim.
