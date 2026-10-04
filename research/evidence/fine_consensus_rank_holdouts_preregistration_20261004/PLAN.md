# Fresh finer-view held-out validation of fixed consensus ranking

Source ALL192 passed298/344TP,4unmatched versus295/4,zero old loss/normal cues,
ALL969 independent head/voter/rank/duplicate/GT replay passed. No E installation.
Run ALL48 INNER first, current installed-source baseline68/80,0unmatched;
strictTP increase/no added unmatched/no old target loss/zero normal cues required.
Only then ALL30 OUTER with current40/56,1unmatched,non-regression and no old loss.
No new thresholds, ranks, heads, GT, model fitting or scene-specific code.

Current prefix comes from accepted paired_pose_native_three_20261004 and original
four-model source/checkpoint outputs, SHA validated against accepted audit.
Fresh original SIFT homography (same reference normal073), original registration
and .85 context gates. For budget room plus at least one existing model row>.05
and reliable SIFT, run actual teacher AND student fine960/stride720/input960
detections; same-weight views count once. Generate same uniform raw7 poses,
three distinct checkpoint SHA consensus,16border,.05floor,old protected prefix.
Use the same accepted9459 head on ORIGINAL observed/registered expected DINO
pixels. Median best-voter-IoU rank within parent, semantic argmax/.98 unchanged,
class-agnostic IoMin.5 exclusion, original5+5 budget and deterministic ordering.

Fresh reliable SIFT is checked before extra detection rather than after it;
unreliable registration would select zero candidates in both orders. Record
every skipped/abstained source and never count abstention as normal success.
All source/expected/photo/model/helper/config pins; only after selection read GT.
One bounded fixed run, cap2Torch/1OpenCV threads and >=6GiB launch RAM.
No competing SAM, no mainline change, no publication of derived/private photos.
Holdout gain alone still needs independent replay and real reference/ROI/Qt/SAM
acceptance. Repeated same-scene development sets are NOT field generalization.
