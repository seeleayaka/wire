# Fixed core-versus-surrounding-ring paired descriptor experiment

Goal: test whether fine-candidate false cues arise from context differences rather
than a changed port body. This is a hypothesis, not an established explanation.
Use ALL192 existing TRAIN sources, audited 9808 curated paired examples and 969
fine proposals. No additional labels, validation fitting, image-ID rules, absolute
position inputs, hyperparameter search, threshold changes or mainline edits.

GT-free descriptor: registered expected image in original source coordinates,
fixed existing bounded global exposure compensation on a COPY only. Extract
the exact component box and a 3x context excluding that box on separate 64x64
grids, with valid-pixel masks and fixed .85 coverage, at least16 core/64 ring
pixels. Fixed 46 scalars: each region has 18 colour/residual/texture statistics,
four core-quadrant residual means, three core-minus-ring residual-channel means,
log aspect ratio, two coverage fractions. No fitted photo exposure gains or
compensation-status flags as classifier inputs. Reference-self examples use
expected pixels as observed, with their existing source-fold membership retained.
Invalid descriptors abort this finite experiment; do not silently filter examples.

Append to frozen 6144 DINO features. Standardize ONLY the 46 added dimensions
using each training fold's mean and std floor .01, clamp [-5,5]. One linear
6190->3 head: seed0,400 steps,AdamW lr.01/decay.001,class-balanced loss.
Same sorted-source-index modulo3 folds including all twins. No neural encoder
training. Classifier-only OOF, not an independently held-out complete detector.

Same .98/three distinct checkpoint SHAs/.85 coverage/5+5 append selector and
old-prefix preservation. Require TRAIN TP>295, unmatched<=4, no lost old targets,
zero normal cues for OOF then full head. Otherwise reject; no full head/held-outs
after failed OOF. Passing TRAIN needs separate fresh INNER strict gain, OUTER
non-regression, real reference/ROI and Qt/SAM acceptance before deployment.

Cache descriptors with pixel/input/script/runtime SHA pins, separately audit
all examples, scores, grouped standardization and selector/GT results. Existing
matched ALL78 original actual endpoint keeps running with frozen original code.
No field accuracy, cross-cabinet or physical-fault confirmation claims.
