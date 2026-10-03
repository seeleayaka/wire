# One cached fine-tile / accepted native classifier experiment

SAHI primary paper https://arxiv.org/abs/2202.06934 supports sliced inference
as a general small-object method; its reported gains are NOT this project's.
WBF primary paper https://arxiv.org/abs/1910.13302 explains multi-model box
fusion. Do not lower this project's gates based on literature performance.

Two raw-seed variants gained2TRAIN and zeroINNER, rejected; keep accepted
295/68/40,4/0/1. The earlier physical960/stride720 fine-tile run used old
paired head238f and two checkpoint gates, source291/8 vs289/4 rejected.
This ONE different trial reuses its actual GT-free teacher/student fine views,
merges with existing distinct checkpoints, SAME7raw poses and >=3uniqueSHA;
uses ACCEPTED native head9459, unchangedp98/.85/onepose-parent/5+5.
Recompute proposal generation against accepted295 prefix (do not reuse old
selection), no training/no synthetic ground-truth boxes. Cached fine views of
same checkpoint remain ONE vote. No threshold or view-size sweep.
TRAIN192 strict gain with no unmatched increase/no old loss/no normal cue
required before anything else. Independent weak-GT coverage is an upper
bound diagnostic only. Existing fine cache lacks INNER/OUTER because prior
TRAIN failed; if source passes, a separately pinned fresh fine holdout run is
needed, then original reference/ROI/live gates before any E integration.
