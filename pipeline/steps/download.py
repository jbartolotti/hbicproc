from pathlib import Path
import shutil
from ._utils import path_exists, run_subprocess


def run(subject, config, dry_run=False):
    output_dir = Path(config["xnat"]["output_dir"]) / subject
    if path_exists(output_dir):
        return {
            "success": True,
            "skipped": True,
            "message": f"Download output already exists at {output_dir}",
            "command": "",
            "returncode": None,
        }

    command = [
        "Rscript",
        config["xnat"]["script_path"],
        "--subject",
        subject,
        "--output-dir",
        str(output_dir),
    ]

    status = run_subprocess(command, dry_run=dry_run)
    if status["success"] and not dry_run:
        status["message"] = f"XNAT download launched for {subject}."
    return status


def flatten_sourcedata_hierarchy(sourcedata_root, dry_run=False):
    """Flatten XNAT-derived sourcedata exports into one DICOM directory per scan.

    This converts structures like:
        sourcedata/006/BL/1-Localizer/resources/DICOM/files/*.dcm
    into:
        sourcedata/006/BL/1-Localizer/*.dcm
    """
    root = Path(sourcedata_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"sourcedata root not found: {sourcedata_root}")

    moved_files = []
    for dicom_path in root.rglob("*.dcm"):
        rel = dicom_path.relative_to(root)
        if "resources" not in rel.parts:
            continue

        resources_index = rel.parts.index("resources")
        if resources_index < 2:
            continue

        scan_dir = root.joinpath(*rel.parts[:resources_index])
        dest_path = scan_dir / dicom_path.name
        if dicom_path.parent == scan_dir:
            continue

        if dry_run:
            moved_files.append((str(dicom_path), str(dest_path)))
            continue

        scan_dir.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            if dest_path.samefile(dicom_path):
                dicom_path.unlink()
                continue
            base = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while True:
                candidate = scan_dir / f"{base}_{counter}{suffix}"
                if not candidate.exists():
                    dest_path = candidate
                    break
                counter += 1

        shutil.move(str(dicom_path), str(dest_path))
        moved_files.append((str(dicom_path), str(dest_path)))

    if not dry_run:
        _cleanup_empty_dirs(root)

    return moved_files


def _cleanup_empty_dirs(path):
    for child in sorted(path.iterdir(), key=lambda p: len(p.parts), reverse=True):
        if child.is_dir():
            _cleanup_empty_dirs(child)
            try:
                child.rmdir()
            except OSError:
                pass
