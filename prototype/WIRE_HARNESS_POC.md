# Wire Harness Segmentation POC

This is an isolated local instance-segmentation experiment. It does not modify
the DINO review entry point, the assembly template, or the rule engine.

## Run

For an interactive local preview, double-click
`启动线束模型能力预览.bat` in the project root. It lets you choose an image,
shows the source, segmentation overlay and an individual mask, then saves the
same evidence artifacts under `output\model_preview\`.

Run one image explicitly with the local model:

```powershell
\.venv\Scripts\python.exe -B prototype\wire_harness_segmentation_poc.py `
  --image "C:\path\to\cabinet\wrong1.png" `
  --output "output\wire_harness_poc\wrong1"
```

The default model is `models\wire_harness_yolov8s_seg_best.pt`. Replacing that
file with another compatible YOLO segmentation `.pt` updates the POC without a
code change. The command stays fully local and writes `report.json`,
`overlay.jpg`, `mask_all_target.png`, and one mask for each of `cable`,
`connector`, `clip`, and `strap`. `AJ20_D6_Cable` and `MHEV_N11_Cable` are
merged into `mask_cable.png`, while their original class names remain in the
report.

The former hosted-model experiment remains available only when explicitly
requested with `--source roboflow` and a `ROBOFLOW_API_KEY`.

## Interpretation boundary

The report is `evidence_only_manual_review`. It records all returned project
labels, while masks are generated only for the four target classes. It does
not claim a cable is missing, correctly connected, or electrically valid.
Before this becomes a Vision Observation input, compare normal and known-wrong
target-device images and check that Cable masks can be skeletonized without
turning rails, labels, or cabinet edges into cable evidence.

No image is uploaded in the default local mode.
