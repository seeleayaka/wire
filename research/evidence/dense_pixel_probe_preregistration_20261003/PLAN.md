# Independent finer localization architecture

Frozen before reading learned coarse-head inner metrics. Dataset-only sampling-grid audit finds134/221 inner crop boxes have a dimension smaller than one32-grid DINO feature token; all576 inner crops are1280px. This motivates a localization architecture change,not changing IoU,scores,labels,heldout images or taking a special-case image rule.

Reuse the existing frozen384x32x32 DINO features and audited656/576 crop/source partition. Add a trainable three-layer RGB stride8 branch16channels and a32channel semantic projection,combine with shared3x3 context64,center/box head at56x56. Learn fine pixel position while keeping DINO semantic backbone unchanged. No filename/absolute coordinate channels. Fixed8 head epochs,AdamW lr.001,decay.01,batch8;fixed last checkpoint only. Existing .5 crop score,.75 source candidate,.5 tile support/two distinct tiles/IoU.5/shared5extra slots remain.

Run independently after the coarse frozen-feature stage completes;preserve all coarse results including failure. Require smoke pixel-gradient and geometry contracts,then crop localization precision>=.90/recall>=.25,then ALL192 training before inner48/outer30 with current277/63/33 baseline,net training/inner gains/no extra unmatched/no old target loss/normal0. No automatic deployment. Small-object resolution correction is a hypothesis,not achieved accuracy or field/generalization evidence.

References: https://github.com/facebookresearch/dinov2 ; https://arxiv.org/abs/1904.07850 ; https://arxiv.org/abs/1904.01355 . These motivate architecture only,not this project's performance.
