# Optional paired median geometry — acceptance evidence, 2026-10-03

Status: accepted as a default-off human-review extension. All four GUI options
remain off by default; there is no fifth switch. The existing fourth enhancement
calls the unchanged paired review first, then conditionally adds median proposals.
The original V3, paired head, models, labels and calibration are preserved.

## Method and boundaries

Same-class weak model boxes require IoU >= 0.5 support from at least two distinct
checkpoint hashes, score > 0.05 and a complete 16-pixel frame margin. A view of
the same checkpoint does not count as an independent model. Use one highest-score
supporter per unique checkpoint; take the median of the four edges, then recheck
support on the resulting box. No transitive component merging, image-name rule,
fixed cabinet position, ground-truth proposal selection or score amplification.

Keep the frozen paired classifier (6144 input dimensions, old DINO encoder,
1.5/3 square context, probability >= 0.98), original normal-reference veto,
ROI/warp-validity gates, exact native-to-hint association and shared 5+5 budget.
Original predictions and review decisions are append-only. Explicit local-ECC
safety abstention is not successful recognition.

## Recorded results

Class-aware IoU >= 0.5 port localization on the existing, reused dataset:

| Group | Original paired | Median extension | Unmatched boxes |
| --- | --- | --- | --- |
| TRAIN 192 | 286/344 | 289/344 | 4, unchanged |
| Inner 48 | 64/80 | 65/80 | 0, unchanged |
| Outer 30 | 33/56 | 33/56 | 1, unchanged |

No old matched targets lost; no new normal-image cues. Independent source audit
reselected/scored all 270 cases and resolved virtual input aliases to actual
hashed artifacts. The portable helper matches all 587 raw cached proposals.

Fresh 22-cohort live workflow passed in 2534.49 seconds: all four new source
cues, all original paired additions, old-cue/risk controls and three fresh normal
controls. There were three explicit local-ECC safety abstentions, not recognition
successes. Original reference, ROI and warp gates were exercised.

New inspection SAM, Agent and actual Qt button/worker/callback passed in
536.69 seconds on source 012: 9/14 to 10/14 port matches, zero unmatched boxes.
Reference SAM was copied from a hash-verified cache; inspection SAM was new.
SAM-pending button blocking, head-drift and geometry-code-drift stale callback
rejection passed. Original cues, images, models and protected hashes unchanged.
Final decision: `possible_difference_manual_review`; Agent awaits human review.
Actual final overlay inspected. Full project regression: 374 passed, zero skipped.

These are dataset localization and bounded workflow results, NOT field accuracy,
cross-cabinet validation, electrical fault confirmation or continuity recognition.

## Evidence and identity

Workspace: `C:/Users/HUAWEI/Documents/Codex/2026-09-20/z`.
Reports under `artifacts/paired_median_current_head_20261003`,
`paired_median_source_audit_20261003`, `paired_median_helper_parity_20261003`,
`paired_median_live_20261003`, `paired_median_complete_sam_20261003/acceptance.json`
and `paired_median_project_regression_20261003`.

Policy: `accepted_paired_preserved_median_geometry_v1_20261003`.
Installed manifest SHA256:
`7bca50fff73d7448bad5c4dc7e370adcab0bd0745108b79b5174dfcfb4e3a582`.
Geometry backend SHA256:
`def85003b459eea6d2185ade90e836763c605796355d1dfd930a31d423367014`.
Portable helper SHA256:
`6e488dea2c324c69b3e9bc1f8420e1fc7ec4806cc956d1d72c3585492fc9414e`.
Classifier SHA256:
`238f517bfdbe5e81f6c98e28c33750fa02de75cf25ec0c93b8876fe72657004a`.

The initial scientific JSON snapshot used CRLF; applying identical JSON as LF
changed byte identity, not parameters. Preserve both snapshots. The installed
manifest and backend binding use the verified LF hash above. The completed SAM
acceptance binds these actual installed bytes; do not replace the older snapshot.

Next uniform pose search remains a workspace-only experiment against the stronger
289/65/33 baseline. No experimental pose rules or weights are deployed here.
