# Validation precision diagnosis and rejected member substitution

Scope: all 30 fixed val01 images, frozen qualified CNN baseline, no test01 reads,
no production changes, no new data or training. Existing 7 user edits preserved.

Three hypotheses checked in order: merge destroys finer geometry; selection hides
already precise merged boxes; original observations lack precise candidates.
The diagnostic exactly reconstructs the current transitive merge groups and
checks their recovered dictionaries against the unchanged merger for every image.
The selected baseline also exactly reproduces the frozen report.

| Candidate geometry pool | Any-overlap source boxes | IoU >= .5 | Fault candidates |
| --- | ---: | ---: | ---: |
| Current selected qualified baseline | 164/245 | 4 | 70 |
| Every merged candidate | 233/245 | 5 | 379 |
| Raw observations in selected groups | 151/245 | 13 | 938 |
| All raw observations | 223/245 | 20 | 1848 |

All-pool/raw rows are unlimited-budget, annotation-assisted diagnostic ceilings,
NOT deployable accuracy or proof of improvement. Searching more boxes raises
chance overlap. Even all raw observations cannot precisely localize 225/245 source
fragments. Damaged class has only 1/143 raw IoU>=.5, vs 0 selected. These labels
are fragmented and may be incomplete, so they are not complete wire-instance truth.

15 precise source matches disappear at merge. Only 1 further precise match is lost
between all-merged and selected. Within selected groups, 9 additional precise
matches exist in raw children but are hidden by parent geometry. This implicates
merging more than final merged-pool ranking, but also exposes an upstream limit.

## Fixed label-free substitution probes, rejected

Keep original anchor; each remaining parent gets one non-edge tile/refinement raw
member. Two predeclared rules: CNN region p95, or different-tile IoU>=.25 support
count followed by CNN p95. No threshold search or per-image branches. Both retain
70 fault and 37 normal candidate slots; each changes 73 boxes across all 30 images.

| Rule | Source overlap | Image overlap | IoU >= .1 | IoU >= .5 | Mean best IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen qualified baseline | 164/245 | 15/15 | 25 | 4 | .03758 |
| Member CNN | 119/245 | 15/15 | 23 | 0 | .02691 |
| Member consensus then CNN | 124/245 | 14/15 | 23 | 2 | .03130 |

Both rejected. Equal candidate count does not mean unchanged normal candidate
geometry or unchanged human burden. No experimental/default CLI was changed.
The existing default-off CNN branch remains frozen at commit 4bd13c2.

## Reproduction and next safe step

```powershell
.\.venv\Scripts\python.exe -B tools/audit_cnn_precision_ceiling.py --output output/a_new_audit/report.json
.\.venv\Scripts\python.exe -B tools/probe_member_precision.py --output output/a_new_probe/report.json
```

Evidence is archived under `output/cnn_precision_diagnosis_val01_20260929/`.
The first member probe originally wrote to its workspace script directory; the
archived tool now requires an explicit fresh output path, with unchanged policy.

Next investigate a two-level review representation: retain parent coverage and
provide bounded, separate within-parent focus hints. Do not call those hints final
fault boxes, hide their count, or use label-selected children in the algorithm.
Validate hint utility and added burden separately on val01 before UI integration.
Precise damaged-wire detection needs better upstream spatial evidence rather than
just re-ranking the present observations. No claim of generalization is established.
