# Fixed geometric augmentation continuation, 2026-10-03

The original coarse and RGB heads failed their unchanged crop feasibility gates.
The RGB head's 23 unmatched predictions contain 18 with same-class IoU <0.1
and 5 with IoU between 0.1 and 0.5; no duplicate or opposite-class IoU>=0.5.
Training loss was still decreasing at epoch8. This motivates a general training
change, not confidence threshold tuning or case-coordinate exceptions.

Reuse the fixed epoch8 RGB/DINO head and verified original train caches.
Train another fixed16 epochs (24 total), AdamW initial lr0.0005 with cosine
decay to0.00005, decay0.01, batch8, clip5. Transform each training example
using a seeded, uniformly chosen dihedral square-grid rotation/reflection.
Transform both RGB and DINO spatial feature maps and original native GT
coordinates consistently; re-encode target centers rather than approximating
offset transformations. These are cached-feature augmentations, NOT fresh
DINO features on rotated images or a proof that DINO is rotation-equivariant.
No augmentation of inner validation. No best-checkpoint selection, score
search, early stopping, changed GT, filename or coordinate classifier channel.

Keep fixed score0.5/P>=0.90/R>=0.25 crop feasibility. Only if it passes, run
the exact existing dense source acceptance against Current V3 (277/63/33),
score0.75, corroborating score0.5, two distinct tiles, IoU0.5, shared5+5,
ALL192 training before48 inner and30 outer. No deployment even on source pass;
fresh reference/Qt checks remain mandatory. Previously seen validation is
exploratory, not an independent generalization estimate or field accuracy.
