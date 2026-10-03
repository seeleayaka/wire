# Assembly reference comparison POC

This throwaway prototype combines three ideas:

1. OpenCV ORB/RANSAC registration, analogous to pattern-matching AOI.
2. A normal-only anomaly map, a lightweight stand-in for Anomalib-style image anomaly localization.
3. An optional Ultralytics YOLO inference hook for a future custom model that identifies connector and memory-slot states.

The input images must show the **same inspection checkpoint**. A picture of a different part of the cabinet is invalid input.

## One-command run

```powershell
.\.venv\Scripts\python.exe prototype\assembly_reference_compare_poc.py --reference "data\raw\图片1.png" --inspection "data\raw\图片2.png" --checklist prototype\checklist.example.json --output output\assembly_poc
```

Use `--reference` repeatedly once multiple verified-good photos exist. Results are human-review prompts, not automatic quality certification.
