# Conservative post-verification experiment, no runtime changes

Freeze accepted295/68/40, unmatched4/0/1. New-proposal tests cannot improve
existing unmatched cues because protected prefixes are kept. Test a separate
precision hypothesis across the COMPLETE population, never selected photos:
the existing paired native classifier identifies an existing cue as OTHER
with probability>=.98. Use same frozen paired contexts1.5/3,.85validity and
original normal-reference/SIFT provenance. Invalid/unreliable contexts retain
old cues, and positive-class disagreement alone never removes anything.
No fitting, threshold sweep, added proposal or relabeling. Record all existing
cues and probabilities; experimental filtered list only, mainline untouched.
TRAIN: three source-held-out native heads must reduce unmatched below4 and
lose ZERO old matched weak targets. ONE frozen full-head replay must also
reduce unmatched and lose none. Then INNER/OUTER must lose no matched targets,
increase no unmatched or normal cue. Accepted prefix/detectors notOOF; this is
classifier-only OOF and repeated same-scene development. Any failure rejects.
No E integration without separate actual-frame/reference/ROI/GUI acceptance;
the original report and rejected-cue evidence must remain available to humans.
