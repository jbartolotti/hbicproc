from pathlib import Path
import csv
import configparser
import json
import re
import shutil
import tempfile
from typing import Dict, List

from pyxnat import Interface


def run(subject, config, dry_run=False, rerun=False):
    output_root = Path(config["xnat"]["output_dir"])
    subject_path = output_root / subject

    if subject_path.exists() and rerun:
        shutil.rmtree(subject_path)

    if subject_path.exists() and not rerun:
        return {
            "success": False,
            "skipped": False,
            "message": (
                f"Download output already exists at {subject_path}. "
                "Use --rerun to overwrite."
            ),
            "command": "",
            "returncode": None,
        }

    command = f"XNAT download for subject {subject} via pyxnat"
    if dry_run:
        return {
            "success": True,
            "skipped": False,
            "message": (
                f"Dry run: would download XNAT data for {subject} to {subject_path}"
            ),
            "command": command,
            "returncode": None,
        }

    try:
        download_subject_from_xnat(subject, config, output_root)
    except Exception as exc:
        return {
            "success": False,
            "skipped": False,
            "message": f"XNAT download failed for {subject}: {exc}",
            "command": command,
            "returncode": None,
        }

    return {
        "success": True,
        "skipped": False,
        "message": f"XNAT download completed for {subject}",
        "command": command,
        "returncode": 0,
    }


def download_subject_from_xnat(subject: str, config: dict, output_root: Path):
    xnat_config = config["xnat"]
    server = xnat_config.get("server")
    project_id = xnat_config.get("project_id")

    if not server or not project_id:
        raise ValueError("xnat.server and xnat.project_id must be set in the pipeline config.")

    credentials_file = xnat_config.get("credentials_file")
    username = xnat_config.get("username")
    password = xnat_config.get("password")
    verify_ssl = xnat_config.get("verify_ssl", True)

    if credentials_file:
        creds = load_xnat_credentials(credentials_file)
        username = username or creds.get("username")
        password = password or creds.get("password")

    if not username or not password:
        raise ValueError(
            "XNAT credentials must be provided via xnat.credentials_file or xnat.username/xnat.password."
        )

    session_map = {}
    session_names_file = xnat_config.get("session_names_file")
    delimiter = xnat_config.get("session_names_delimiter")
    if session_names_file:
        session_map = load_session_map(session_names_file, delimiter=delimiter)

    subject_id = normalize_subject_label(subject)
    temp_root = Path(tempfile.mkdtemp(prefix=f"hbicproc_xnat_{subject_id}_"))
    try:
        interface = Interface(server=server, user=username, password=password, verify=verify_ssl)
        try:
            experiments = list_subject_experiments(interface, project_id, subject_id)
            if not experiments:
                raise ValueError(f"No XNAT sessions found for subject '{subject}'.")

            download_plan = build_download_plan(subject_id, experiments, session_map)
            if not download_plan:
                raise ValueError(
                    f"Unable to derive any session labels for {subject}." 
                    " Check xnat.session_names_file or session naming conventions."
                )

            subject_temp_dir = temp_root / subject
            subject_temp_dir.mkdir(parents=True, exist_ok=True)

            for exp_label, bids_session in download_plan.items():
                session_temp_dir = subject_temp_dir / bids_session
                download_experiment(interface, project_id, subject_id, exp_label, session_temp_dir)
                verify_downloaded_session(session_temp_dir)

            subject_output_dir = output_root / subject
            subject_output_dir.mkdir(parents=True, exist_ok=True)
            for session_dir in sorted(subject_temp_dir.iterdir()):
                destination = subject_output_dir / session_dir.name
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.move(str(session_dir), str(destination))
        finally:
            try:
                interface.disconnect()
            except Exception:
                pass
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def normalize_subject_label(subject: str) -> str:
    subject = subject.strip()
    if subject.lower().startswith("sub-"):
        return subject[4:]
    return subject


def normalize_session_label(session_value: str, subject_id: str) -> str:
    if not session_value:
        raise ValueError("Session value is empty; cannot derive BIDS session label.")

    session_token = session_value.strip()
    if subject_id:
        session_token = re.sub(
            rf"^(sub[-_]?){re.escape(subject_id)}[_-]?",
            "",
            session_token,
            flags=re.IGNORECASE,
        )
        session_token = re.sub(
            rf"^{re.escape(subject_id)}[_-]?",
            "",
            session_token,
            flags=re.IGNORECASE,
        )

    session_token = session_token.strip()
    if session_token.lower().startswith("ses-"):
        session_token = session_token[4:]

    session_token = session_token.replace("_", "-")
    session_token = session_token.strip(" -")
    if not session_token:
        raise ValueError(f"Cannot normalize session label from '{session_value}'.")

    return f"ses-{session_token}"


def load_xnat_credentials(credentials_file: str) -> Dict[str, str]:
    path = Path(credentials_file)
    if not path.exists():
        raise FileNotFoundError(f"XNAT credentials file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"XNAT credentials file is empty: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return {
            "username": data.get("username") or data.get("user"),
            "password": data.get("password") or data.get("pass"),
        }

    parser = configparser.ConfigParser()
    try:
        parser.read_string(text)
    except configparser.Error:
        parser = None

    if parser and parser.sections():
        section = parser[parser.sections()[0]]
        return {
            "username": section.get("username") or section.get("user"),
            "password": section.get("password") or section.get("pass"),
        }

    result = {"username": None, "password": None}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in {"username", "user"}:
                result["username"] = value
            elif key in {"password", "pass", "pwd"}:
                result["password"] = value
    if result["username"] and result["password"]:
        return result

    raise ValueError(
        f"Unable to parse XNAT credentials from {path}. "
        "Use JSON or key=value pairs with username and password."
    )


def load_session_map(session_names_file: str, delimiter: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    path = Path(session_names_file)
    if not path.exists():
        raise FileNotFoundError(f"Session names file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"Unable to parse session names file: {path}")

        entries = {}
        for row in reader:
            xnat_label = row.get("session_sic") or row.get("xnat_label") or row.get("label")
            session_value = row.get("session") or row.get("session_id") or row.get("ses")
            subject_value = row.get("subject") or row.get("subject_id") or row.get("sub")
            if not xnat_label or not session_value:
                continue

            session_value = session_value.strip()
            if not subject_value and delimiter and delimiter in session_value:
                parts = session_value.split(delimiter, 1)
                if len(parts) == 2:
                    subject_value, session_value = parts

            entries[xnat_label.strip()] = {
                "subject": normalize_subject_label(subject_value) if subject_value else None,
                "session_value": session_value.strip(),
            }

    return entries


def list_subject_experiments(interface: Interface, project_id: str, subject_id: str) -> List[str]:
    subject_obj = interface.select.project(project_id).subject(subject_id)
    if not subject_obj.exists():
        raise ValueError(f"XNAT subject '{subject_id}' not found in project '{project_id}'.")

    experiments = subject_obj.experiments()
    labels = _selection_get(experiments)
    if labels:
        return labels

    fallback = interface.select(f"/projects/{project_id}/subjects/{subject_id}/experiments/*")
    return _selection_get(fallback)


def build_download_plan(subject_id: str, experiment_labels: List[str], session_map: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    plan = {}
    sessions_seen = set()
    for exp_label in sorted(experiment_labels):
        if exp_label in session_map:
            entry = session_map[exp_label]
            if entry["subject"] and entry["subject"] != subject_id:
                continue
            session_label = normalize_session_label(entry["session_value"], subject_id)
        else:
            session_label = normalize_session_label(exp_label, subject_id)

        if session_label in sessions_seen:
            raise ValueError(
                f"Duplicate BIDS session label '{session_label}' derived from XNAT labels: {exp_label}."
            )
        sessions_seen.add(session_label)
        plan[exp_label] = session_label

    return plan


def download_experiment(interface: Interface, project_id: str, subject_id: str, experiment_label: str, destination: Path):
    exp = interface.select.project(project_id).subject(subject_id).experiment(experiment_label)
    destination.mkdir(parents=True, exist_ok=True)

    resources = exp.resources()
    resource_names = _selection_get(resources)
    if not resource_names:
        raise ValueError(f"No resources found for experiment '{experiment_label}'.")

    for resource_name in sorted(resource_names):
        resource = exp.resource(resource_name)
        files = resource.files()
        file_names = _selection_get(files)
        if not file_names:
            continue

        resource_dir = destination / resource_name
        resource_dir.mkdir(parents=True, exist_ok=True)
        for file_name in sorted(file_names):
            target_file = resource_dir / file_name
            file_obj = resource.file(file_name)
            if hasattr(file_obj, "get_copy"):
                file_obj.get_copy(str(target_file))
            else:
                file_obj.get(str(target_file))


def verify_downloaded_session(session_dir: Path):
    if not session_dir.exists():
        raise FileNotFoundError(f"Downloaded session directory not present: {session_dir}")

    if not any(session_dir.rglob("*")):
        raise ValueError(f"Downloaded session directory is empty: {session_dir}")


def _selection_get(selection) -> List[str]:
    if selection is None:
        return []

    if hasattr(selection, "get"):
        items = selection.get()
    else:
        items = list(selection)

    if isinstance(items, str):
        return [items]
    if items is None:
        return []
    return list(items)


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
