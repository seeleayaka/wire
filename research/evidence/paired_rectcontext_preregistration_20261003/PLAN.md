# True candidate-shaped visual context instead of square or token footprint

Independent train-only architecture control, not accepted code. Existing
square max-axis crop hides short-axis geometry; sixteen-grid footprint pooling
and scalar/narrow MLP controls did not yield accepted accuracy. A rectangular
pixel crop changes the DINO input itself, so narrow width/height information
need not be recovered from a coarse16-token grid.

At existing scales1.5 and3, use centered width=max(4,box_width*scale),
height=max(4,box_height*scale). Same floor/ceil borders, padding(124,116,104),
BGR->RGB,224x224 INTER_AREA and ImageNet normalization. This anisotropic
normalization changes apparent shape; it is a testable risk, NOT known gain.
Keep frozen same DINO CLS and central4x4 feature vector1536/paired6144,
normalization and observed/expected/absdiff/product. Same original TRAIN192
groups/344GT and all4166 existing rows (original3134 plus1032 uniform aspect
training examples). Reference-self vectors are expected/expected. No IDs, XY,
GT IoU scalars or aspect scalar channels. Coverage must pass original square
.85 context too (stronger than contained rectangular context); no gate relaxed.
Exact identical rectangular floor/ceil crops deduplicated, no square vectors
reused. Smoke first2 GT sources must prove real finite descriptor differences
and immutable encoder/runtime before full extraction.

Full: original fixed400 linear/AdamW.01/.001/source-fold3/seed0/class balancing/
p.98/P.98/R.25; actual source gain vs accepted full286/344/4 then exact full
head TRAIN and fresh inner48vs64/80/0 and outer30vs33/56/1. Any failed gate
stops, no automatic deployment. If aspect-augmented crop gate rejects, one
predeclared matched3134-row control may test removal of ONLY synthetic_aspect
without new feature extraction or threshold/optimizer tuning. Source groups
and untouched same344GT retained. No validation training or selected-best
checkpoint. GT only training labels and post-selection scoring. Source pass
requires original reference/ROI/SAM/Qt verification; all validation reused,
not field/cross-cabinet accuracy. Independent positive-jitter run preserved.
