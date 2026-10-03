"""Build the minimal transferable cable-cabinet review bundle.

The result contains only the DINO+SAM3 operator route, its local models, a
Windows x64 runtime, the requested cabinet cases, and clean Linux bootstrap
sources.  It deliberately excludes all credentials, histories, datasets,
experiments, caches, outputs and training material.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CASES = Path(r"C:\Users\HUAWEI\Desktop\案例\线材柜子")
WINDOWS_PYTHON = Path(r"C:\Users\HUAWEI\AppData\Local\Programs\Python\Python311")
MAIN_SITE = ROOT / ".venv" / "Lib" / "site-packages"
SAM3_SITE = ROOT / "runtime" / "sam3" / ".venv" / "Lib" / "site-packages"
SAM3_BASE_SITE = WINDOWS_PYTHON / "Lib" / "site-packages"
ENTRY_MODULE = "assembly_auto_review_dino_v2"


def _ignore_names(_directory: str, names: list[str]) -> set[str]:
    skipped = {".git", ".github", "__pycache__", "tests", "test", "docs", "examples", "notebooks", "build"}
    return {
        name
        for name in names
        if name in skipped
        or name.endswith((".pyc", ".pyo", ".whl"))
        or name.startswith(".pytest_cache")
    }


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Required directory is missing: {source}")
    shutil.copytree(source, destination, ignore=_ignore_names)


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _module_path(module: str) -> Path | None:
    prototype = ROOT / "prototype"
    candidate = prototype / (module.replace(".", "/") + ".py")
    if candidate.is_file():
        return candidate
    package_init = prototype / module.replace(".", "/") / "__init__.py"
    return package_init if package_init.is_file() else None


def _core_modules() -> list[Path]:
    pending = [ENTRY_MODULE]
    visited: set[str] = set()
    selected: set[Path] = set()
    # These are reached through compatibility aliases at runtime and are kept
    # explicit so the portable route does not depend on historical scripts.
    pending.extend(
        [
            "assembly_auto_review_dino",
            "assembly_auto_review_robust_v6",
            "assembly_auto_review_robust_v5",
            "assembly_auto_review_robust_v4",
            "assembly_auto_review_robust_v3",
            "assembly_auto_review_robust_v2",
            "assembly_auto_review_robust",
            "assembly_auto_review_multiscale",
            "assembly_auto_review_results",
            "assembly_auto_review",
            "assembly_template_review_app",
            "assembly_multi_template_review_color",
            "assembly_anchor_review_app",
            "dimm_review_app_fixed",
            "dimm_upper_right_review",
            "dino_feature_diff_v2",
            "dino_feature_diff",
            "tiled_dino_review",
            "deepseek_mask_review",
            "sam3_wire_fusion",
        ]
    )
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = _module_path(module)
        if path is None:
            continue
        selected.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: Iterable[str]
            if isinstance(node, ast.Import):
                names = (item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = (node.module,)
            else:
                continue
            for name in names:
                if _module_path(name) is not None:
                    pending.append(name)
    return sorted(selected)


def _copy_portable_python(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in WINDOWS_PYTHON.iterdir():
        if item.name in {"Lib", "Doc", "Tools", "include", "libs", "Scripts", "tcl", "share"}:
            continue
        if item.is_file():
            _copy_file(item, destination / item.name)
        elif item.is_dir():
            _copy_tree(item, destination / item.name)
    standard_library = WINDOWS_PYTHON / "Lib"
    for item in standard_library.iterdir():
        if item.name == "site-packages":
            continue
        if item.is_dir():
            _copy_tree(item, destination / "Lib" / item.name)
        else:
            _copy_file(item, destination / "Lib" / item.name)


def _copy_site_items(source_root: Path, destination_root: Path, item_names: Iterable[str]) -> None:
    """Copy exactly the importable packages and metadata required by one runtime."""
    for name in item_names:
        source = source_root / name
        if not source.exists():
            raise FileNotFoundError(f"Required site-package item is missing: {source}")
        destination = destination_root / name
        if source.is_dir():
            _copy_tree(source, destination)
        else:
            _copy_file(source, destination)


MAIN_SITE_ITEMS = (
    "PyQt5",
    "PyQt5-5.15.11.dist-info",
    "PyQt5_Qt5-5.15.2.dist-info",
    "PyQt5_sip-12.19.0.dist-info",
    "cv2",
    "opencv_python-5.0.0.93.dist-info",
    "numpy",
    "numpy.libs",
    "numpy-2.4.6.dist-info",
    "torch",
    "torchgen",
    "functorch",
    "torch-2.13.0.dist-info",
    "filelock",
    "filelock-3.32.3.dist-info",
    "fsspec",
    "fsspec-2026.7.0.dist-info",
    "jinja2",
    "jinja2-3.1.6.dist-info",
    "markupsafe",
    "markupsafe-3.0.3.dist-info",
    "mpmath",
    "mpmath-1.3.0.dist-info",
    "networkx",
    "networkx-3.6.1.dist-info",
    "sympy",
    "sympy-1.14.0.dist-info",
    "typing_extensions.py",
    "typing_extensions-4.16.0.dist-info",
)


def _write_configurations(bundle: Path) -> None:
    config = bundle / "config"
    config.mkdir(parents=True, exist_ok=True)
    dino_recipe = json.loads((ROOT / "config" / "cabinet_dino_review_recipe.json").read_text(encoding="utf-8"))
    dino_recipe["reference_image"] = "data/test_cases/line_cabinet/4/right.png"
    (config / "cabinet_dino_review_recipe.json").write_text(
        json.dumps(dino_recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sam3_recipe = json.loads((ROOT / "config" / "cabinet_sam3_fusion_recipe.json").read_text(encoding="utf-8"))
    sam3_recipe["sam3_python"] = {
        "windows": "runtime/windows/python/python.exe",
        "linux": "runtime/sam3/.venv/bin/python",
    }
    sam3_recipe["sam3_site_packages"] = {"windows": "runtime/windows/sam3_site_packages"}
    sam3_recipe["sam3_base_site_packages"] = {"windows": "runtime/windows/sam3_base_site_packages"}
    sam3_recipe["sam3_runner"] = "runtime/sam3/run_sam3_cable_probe.py"
    sam3_recipe["cache_dir"] = "data/runtime_cache/sam3"
    (config / "cabinet_sam3_fusion_recipe.json").write_text(
        json.dumps(sam3_recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _copy_file(ROOT / "config" / "deepseek_mask_review.json", config / "deepseek_mask_review.json")
    _copy_file(ROOT / "config" / "deepseek_mask_review.local.example.json", config / "deepseek_mask_review.local.example.json")
    _copy_tree(ROOT / "config" / "topology_demo", config / "topology_demo")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(bundle: Path) -> None:
    records = []
    for path in sorted(path for path in bundle.rglob("*") if path.is_file()):
        records.append(
            {
                "path": path.relative_to(bundle).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "package": "CableCabinetReviewPortable",
        "built_on": date.today().isoformat(),
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "files": records,
    }
    (bundle / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_readme(bundle: Path) -> None:
    (bundle / "PACKAGE_README.md").write_text(
        """# Line Cabinet Review Package

This package contains only the current cable-cabinet review route: image
alignment, DINO difference candidates, SAM3 visible-cable evidence, optional
DeepSeek mask review, an auditable inspection-Agent task workflow, the required
local models, and the supplied cabinet cases. The Agent keeps machine evidence
separate from human conclusions, records repair guidance only after explicit
confirmation, and can append reinspection evidence to the same task. It does
not contain API keys, historical outputs, training datasets, experiments,
source-control data, or test code.

## Windows x64

1. Extract the ZIP completely before starting it.
2. Double-click `0_first_setup_and_start.bat` once. It verifies the bundled
   runtime and opens the application. No Python installation or package download
   is required.
3. Later, double-click `启动装配自动定位复核_DINO融合版.bat`.

The package is tested for Windows x64. It uses CPU inference and needs several
GB of free disk space for new result and SAM3 cache files.

## Linux x64

Install Python 3.11 plus your desktop system's Qt/OpenCV libraries, then run:

```bash
bash runtime/bootstrap.sh --recreate
bash start_cable_review.sh
```

Linux must create its own native binary environment; the bundled Windows
runtime cannot be reused across operating systems. The Linux bootstrap installs
the pinned packages from the internet on first use. Local model files are
already in this package.

## Cases and DeepSeek

The supplied cabinet cases are in `data/test_cases/line_cabinet/`. The default
reference is `4/right.png`. DeepSeek is optional. Enter a key in the GUI only
when needed; it is saved locally on that computer and is deliberately not part
of this transfer package.
""",
        encoding="utf-8",
    )


def _create_zip(bundle: Path, zip_path: Path) -> None:
    binary_suffixes = {".dll", ".exe", ".pyd", ".pt", ".pth", ".png", ".jpg", ".jpeg", ".zip", ".gz", ".whl"}
    files = sorted(path for path in bundle.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as archive:
        for index, path in enumerate(files, 1):
            compression = zipfile.ZIP_STORED if path.suffix.lower() in binary_suffixes or path.stat().st_size >= 16 * 1024 * 1024 else zipfile.ZIP_DEFLATED
            archive.write(path, arcname=(Path(bundle.name) / path.relative_to(bundle)).as_posix(), compress_type=compression)
            if index % 1000 == 0 or index == len(files):
                print(f"ZIP: {index}/{len(files)} files")


def build(output_root: Path, make_zip: bool) -> tuple[Path, Path | None]:
    bundle = output_root / "CableCabinetReviewPortable"
    if bundle.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {bundle}")
    if not CASES.is_dir():
        raise FileNotFoundError(f"Cabinet cases are missing: {CASES}")
    for required in (WINDOWS_PYTHON, MAIN_SITE, SAM3_SITE, SAM3_BASE_SITE):
        if not required.exists():
            raise FileNotFoundError(f"Required Windows runtime source is missing: {required}")

    print("Copying operator source...")
    for source in _core_modules():
        _copy_file(source, bundle / "prototype" / source.name)
    _copy_tree(ROOT / "inspection_agent", bundle / "inspection_agent")
    _copy_file(ROOT / "tools" / "inspection_agent_cli.py", bundle / "tools" / "inspection_agent_cli.py")
    print("Copying startup and bootstrap scripts...")
    for relative in (
        Path("0_first_setup_and_start.bat"),
        Path("启动装配自动定位复核_DINO融合版.bat"),
        Path("start_cable_review.sh"),
        Path("runtime/bootstrap.ps1"),
        Path("runtime/bootstrap.sh"),
        Path("runtime/requirements-main.txt"),
        Path("runtime/sam3/requirements.txt"),
    ):
        _copy_file(ROOT / relative, bundle / relative)
    print("Copying local DINO and SAM3 model assets...")
    _copy_tree(ROOT / "models" / "dinov2" / "dinov2", bundle / "models" / "dinov2" / "dinov2")
    _copy_file(ROOT / "models" / "dinov2" / "hubconf.py", bundle / "models" / "dinov2" / "hubconf.py")
    _copy_file(ROOT / "models" / "dinov2" / "weights" / "dinov2_vits14_pretrain.pth", bundle / "models" / "dinov2" / "weights" / "dinov2_vits14_pretrain.pth")
    _copy_file(ROOT / "models" / "sam3" / "sam3.pt", bundle / "models" / "sam3" / "sam3.pt")
    _copy_tree(ROOT / "runtime" / "sam3" / "source" / "sam3", bundle / "runtime" / "sam3" / "source" / "sam3")
    _copy_file(ROOT / "manual_review" / "run_sam3_cable_probe.py", bundle / "runtime" / "sam3" / "run_sam3_cable_probe.py")
    print("Copying requested cabinet cases...")
    _copy_tree(CASES, bundle / "data" / "test_cases" / "line_cabinet")
    print("Creating portable Windows runtime...")
    _copy_portable_python(bundle / "runtime" / "windows" / "python")
    _copy_site_items(MAIN_SITE, bundle / "runtime" / "windows" / "main_site_packages", MAIN_SITE_ITEMS)
    _copy_tree(SAM3_SITE, bundle / "runtime" / "windows" / "sam3_site_packages")
    _copy_tree(SAM3_BASE_SITE, bundle / "runtime" / "windows" / "sam3_base_site_packages")
    _write_configurations(bundle)
    _write_readme(bundle)
    key_path = bundle / "config" / "deepseek_mask_review.local.json"
    if key_path.exists():
        raise RuntimeError("Refusing to package a local DeepSeek API key.")
    _write_manifest(bundle)
    zip_path = output_root / "CableCabinetReviewPortable.zip" if make_zip else None
    if zip_path is not None:
        print("Creating ZIP64 archive...")
        _create_zip(bundle, zip_path)
    return bundle, zip_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--zip-existing", action="store_true", help="Create only the ZIP64 archive from an already validated staging directory.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.zip_existing:
        bundle = args.output / "CableCabinetReviewPortable"
        archive = args.output / "CableCabinetReviewPortable.zip"
        if not bundle.is_dir():
            raise FileNotFoundError(f"Existing staging directory is missing: {bundle}")
        if archive.exists():
            raise FileExistsError(f"Refusing to overwrite existing ZIP64 archive: {archive}")
        if (bundle / "config" / "deepseek_mask_review.local.json").exists():
            raise RuntimeError("Refusing to package a local DeepSeek API key.")
        print("Creating ZIP64 archive from existing staging directory...")
        _create_zip(bundle, archive)
        print(f"ZIP64: {archive}")
        return
    bundle, archive = build(args.output, make_zip=not args.no_zip)
    print(f"Bundle: {bundle}")
    if archive is not None:
        print(f"ZIP64: {archive}")


if __name__ == "__main__":
    main()
