# Prototype verdict — 2026-08-17

The two sample images can be feature-aligned: the prototype found 675 good
matches and 356 RANSAC inliers. However, full-image pixel comparison produced
75 difference regions, including several large regions. It is therefore not
suitable as a direct cable-error decision.

Validated next design: guided, per-check-item close-up comparison. Each
connector group needs its own reference crop and its own human-review result.
