# One bounded photometric proposal trial; no deployment

Written before detector results. Motivation: illumination can change detector
inputs even when geometry is unchanged. The separately verified TRAIN192
photometric estimator is a hypothesis, not recognition acceptance.

Keep original source/reference, original DINO paired features, accepted 9459
classifier and all current native baseline cues untouched. Use compensation
ONLY for a separate detector proposal copy, and only when its fixed policy
returns compensated (identity/abstention produce no new branch). No transformed
photos saved/published. Run the same original1280/stride960,960-input tiled
predictor on three different accepted checkpoint SHAs. Same seven generic poses,
median seed construction, unique SHA votes>=3, p>=.98, context validity>=.85,
5 primary+5 shared extra budget. Views of one weight never extra votes.

Source metadata identifies original coordinate geometry; record transformed
pixel SHA separately and never claim the original pixels were inferred. A
two-source smoke uses alphabetic eligible TRAIN sources, never positive labels;
no metrics in smoke. Full trial needs smoke success and independent estimator
replay. No parameter search or classifier fitting. Select before reading labels.

Full source gates: ALL192 TRAIN strictly more TP than accepted295/344 with
unmatched<=4, no old loss and zero normal additions; only then ALL48 INNER
strictly more than68/80 with unmatched0; only then ALL30 OUTER at least40/56
with unmatched<=1. Existing validation is reused development data, not fresh
generalization. Training detector/prefix already saw TRAIN. Stop at first failed
gate. Even a source pass requires actual original-reference/ROI filtering and
all-normal/control/changed-input GUI/SAM acceptance before any E change. Keep
E DEFAULT OFF and all installed model/config/backend bytes unchanged.

Bounded CPU two threads and offline model inference. Parallel computation is
allowed only with at least6GiB available RAM; no SAM recomputation, external
training, paid services, purchases or quota reset. All inputs/runtime SHA pinned.
