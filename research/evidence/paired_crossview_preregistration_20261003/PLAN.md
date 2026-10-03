# Context plus candidate-footprint agreement, fixed gates

Standalone footprint controls fail: original3134 rows214correct/5errors,
P97.7169%,GT211/34461.337%; full4166 nonlinear359/32,P91.8159%,GT216/34462.79.
Do not roundP97.72 up to98 or deploy failed standalone heads. The original
context semantic head has cropP98.65 but missed localization (inner wire FP),
while footprint head explicitly averages candidate extents. Test their
conjunction as a NEW distinct fixed policy, not a standalone reversal.

Both class argmax must agree AND both probabilities>=.98. Min member score
is an explicit manual review score, NOT calibrated physical-fault probability
or independent evidence. Same frozenDINO sharedbody/correlateddetectors and
source data; disclose dependence. Original context head3OOF trained1107;
footprint head3OOF trained3134. Evaluate ALL same3134 samples, preserving
source groups; context embeddings re-index by exact sample identity/kind/box/
label/sourcefold, not image metadata to network. Original344GT denominator,
P98/R25, no class-specific exceptions, score tuning or best checkpoint.

If crop passes, ALL192 actual native weak proposals with sameOOF head pairs
add after accepted geometry286/344/4, shared5+5. NeedTPgain, no new unmatched,
oldloss/normalcue0. Fail stops. After trainpass fit one fixed400-step full
footprint linear head3134; original semantic full3b52420e stays immutable.
Exact full pair TRAIN192 must independently pass. Then fresh48inner vs64/80/0
requires gain/no error,30outer vs33/56/1 nonregression. Original proposalfloor/
IoU/warp/contextreference thresholds unchanged. No newfeaturetypes/models
or training after heldouts, no GT selection or automatic deployment.

Even all pass requires actual reference/ROI/SAM/Qt. Existing validation was
reused, not independent field/continuity/crosscabinet evidence. Current
accepted default-off geometry branch and all old cues stay intact.
