# Plain-training control for the failed D4 continuation

Epoch8 RGB/DINO head crop P84.03%/R54.75% failed90% gate. Additional16 cached
feature D4 epochs reduced P78.29%/R45.70%. A separate source relocalizer with
epoch8 head added0 on ALL192;40/50 proposal regions produced no strict head
output in either context. Distinguish insufficient head training from harmful
cached-feature rotation using a plain control, not a confidence-threshold edit.

Same fixed epoch8 checkpoint; another fixed16 epochs,24 total; same AdamW
lr0.0005 cosine to0.00005, decay0.01, batch8, clip5, source-disjoint192/48,
same train-cache order seeds and no geometric/photometric augmentation.
Unchanged architecture, class labels, IoU0.5, score0.5/P90/R25 crop feasibility.
Fixed last checkpoint; do not select a best epoch or tune thresholds.
Only crop pass launches ALL192 train/48 inner/30 outer original independent
dense source gate with candidate0.75,two tiles0.5,IoU0.5,Current V3 prefix,
shared5+5. No auto-deployment. Validation reused, not independent accuracy.
