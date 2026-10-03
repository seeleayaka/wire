# Fixed SAM visible-geometry audit and one-image validation pilot

Frozen before examining the new pilot result. No test01 reads or tuning.

- Cache feasibility audit: existing train01 normal_003, damaged_001,
  disconnected_001, prompt cable, threshold 0.4. Per-instance components >=50
  pixels, preserve masks independently; no joining fragments into cable IDs.
- Guard: one connected skeleton, exactly two tips, zero branch clusters.
  Branch/loop/disconnected ambiguity abstains; no pruning to force acceptance.
- Pilot: val01 disconnected_002.JPG (fixed first disconnect example), production
  local SAM3 checkpoint, CPU 8 threads, existing resolution 1008 and bf16 runner,
  cable, 0.4, fresh output and fresh image encoder. No prompt/threshold sweeps.
- Source rectangles are scoring-only fragmented defect annotations, not wire-mask
  truth. Report any-mask intersection and all raw component-bbox IoU geometric
  ceiling in original source coordinates. Do not select components using labels.
- A single fault image has no normal false-positive estimate and cannot establish
  validation-set or generalization improvement. Preserve coarse CNN/DINO path.
- Historical caches have no original source/checkpoint fingerprints: current
  checksums cannot prove historical identity. New pilot records current source,
  checkpoint and runner hashes, but historical/current masks are not a paired
  defect detector. No cable identity, continuity, automatic electrical fault.

Coordinate diagnostic: capture the actual production SIFT homography with seed 0,
transform scoring rectangles and compare them to the frozen val cache, warp masks
with that same H and valid-source mask. Generate all component bounds >=50 pixels
and all such component bounds inside each frozen parent; no scores, label-based
selection, padding, pruning or cap. This is a geometric ceiling including normal
wire regions, not an implementable fault ranking or a budget-matched improvement.
