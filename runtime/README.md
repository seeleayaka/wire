# Portable Runtime Layout

The main cable review pipeline keeps all required model assets beneath this
project directory:

- `models/dinov2/`: local DINOv2 source and ViT-S/14 checkpoint.
- `models/sam3/sam3.pt`: local SAM3 checkpoint.
- `runtime/sam3/source/`: local SAM3 source checkout.
- `.venv/`: main GUI environment for this computer.
- `runtime/sam3/.venv/`: isolated SAM3 worker environment for this computer.
- `assets/cabinet_front/right.png`: default cabinet reference image.

The copied virtual environments are usable on this computer, but Windows and
Linux venvs embed their original base-Python location and are not portable to
another user profile, computer, or operating system. Rebuild them in place on
the destination computer.

On Windows x64, install Python 3.11 and make sure `python --version` reports
`3.11`. For a transferred ZIP, double-click `0_first_setup_and_start.bat` at
the project root. It rebuilds both project-local environments, then starts the
review UI. This requires package access to PyPI/PyTorch wheels once.

The equivalent manual PowerShell command is:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\runtime\bootstrap.ps1 -Recreate
```

`-Recreate` only replaces the two virtual environments inside this project. It
does not delete models, source code, reference assets, settings, or output.
After that first setup, use `启动装配自动定位复核_DINO融合版.bat` normally.
Model inference itself uses only the bundled local model files.

On Linux x64 with a graphical desktop, install Python 3.11 plus the system
packages needed by Qt/OpenCV. On Ubuntu/Debian, this is typically:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv libgl1 libglib2.0-0 libxkbcommon-x11-0 libxcb-cursor0
```

Then run from the project root:

```bash
bash runtime/bootstrap.sh --recreate
bash start_cable_review.sh
```

Set `PYTHON_BIN` before the bootstrap command when Python 3.11 is installed
under a nonstandard name, for example `PYTHON_BIN=/usr/local/bin/python3.11
bash runtime/bootstrap.sh --recreate`. The Linux script deliberately replaces
the copied Windows virtual environments only when `--recreate` is supplied.

DeepSeek remains optional. Do not package `config/deepseek_mask_review.local.json`;
create it from the example or set `DEEPSEEK_API_KEY` on the destination computer.
