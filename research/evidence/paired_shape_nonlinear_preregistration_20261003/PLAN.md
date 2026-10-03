# Fixed nonlinear shape/context interaction control

Shape crop-identifiability audit confirmed1032 exact crop matches and666
contradictory TRAINIoU labels. Linear6147 features fixedOOF yieldedP89.72%,
GTrecall7.85%, failed both fixed gates. Preserve this rejection; no source
or heldout claims. Test whether missing nonlinear context/shape interaction,
not candidate threshold, is the blocker.

Same exact4166 rows,paired6144+3normalized shape,sourcefolds3,seed0,
inverse-frequency class weighting,AdamW.01/.001,fixed400wholebatch steps,
.98 probability/P.98/GTrecall.25. Replace linear by6147->32GELU->3 only.
First linear standard initialization, final layer zeros. No best checkpoint,
epoch/threshold sweep, heldout labels or encoder inference. Single CPU thread.
Finite-gradient smoke before fitting; rejected crop means no source gate.
OOF source stage adds only after fixed geometry286/344/4, old5+5/prefix kept.
Must gainTP with no unmatched,normal0/oldloss0; then full and heldouts later.
No release/SAM modifications, no deployment or independent accuracy claim.
