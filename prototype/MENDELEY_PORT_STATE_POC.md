# Mendeley fixed-Dell visible port-state POC

This POC detects only visibly unplugged plugs and visibly empty/disconnected
jacks in the Mendeley Dell chassis.  It is not a cable-topology or electrical
continuity system.

The source dataset has opaque class IDs and no `data.yaml`.  The mapping below
was established by rendering and manually reviewing source labels:

| Source ID | Meaning | Used here |
| --- | --- | --- |
| 1 | damaged cable | no |
| 2 | misrouted cable | no |
| 3 | disconnected plug | yes: `unplugged_plug` |
| 4 | disconnected jack | yes: `unplugged_jack` |

The preparation tool retains the supplied train/validation/test split, converts
only IDs 3 and 4 into rectangular segmentation polygons, and creates a new
derived dataset.  Its images are hard-linked where possible; source data is
never altered.

```powershell
E:\PythonProject10\.venv\Scripts\python.exe -B tools\prepare_mendeley_port_state_poc.py `
  --dataset-root 'E:\PythonProject10\data\external_datasets\mendeley_electrical_wiring_faults\Predictive Maintenance for Electrical Wiring Faults' `
  --output E:\PythonProject10\data\derived\mendeley_port_state_rectseg_20260825
```

The CPU training gate is deliberately short and does not establish field
accuracy.  It uses the already-local `yolov8s-seg.pt` because this computer has
no local generic detection base weight.

```powershell
E:\PythonProject10\.venv\Scripts\python.exe -B tools\train_mendeley_port_state_poc.py `
  --data E:\PythonProject10\data\derived\mendeley_port_state_rectseg_20260825\mendeley_port_state_rectseg.yaml
```

The inference program returns `possible_unseated_port_manual_review` for a
candidate, otherwise `no_unseated_port_candidate_not_verified`.  Neither result
is an electrical or complete routing verdict.
