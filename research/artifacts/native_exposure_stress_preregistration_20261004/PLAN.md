# Fixed whole-validation exposure robustness audit

Several extra proposal variants failed strict net-gain gates; accepted E
native295/68/40,4/0/1 unchanged. Do not keep tuning a handful of source images.
Use ALL48INNER+30OUTER images, including normal and non-disconnection controls,
same photo-independent inspection transform RGB*0.85 rounded/clipped to8bit,
losslessPNG. Reference remains ORIGINAL known normal073, unchanged SHA.
No geometric change: original weak box labels remain at original coordinates.
NO fitting or parameter sweep, synthetic illumination variation not new scenes.
Each image reruns actual InitialReviewWorker SIFT/DINO and accepted median /
native backend with frozen models. Do not reuse old boxes/matrices for changed
pixels. Compare fresh median vs fresh native and both with original reviewed
development counts, per source/whole cohort. Record original source SHA,
transformed source SHA, exposure constant, every fallback and all weak-GT
metrics. Safety abstention isn't successful detection. No SAM recomputation
or electrical continuity/fault claim; this is core-port robustness diagnostics.
No E changes and no publication of source or generated dataset photos.
