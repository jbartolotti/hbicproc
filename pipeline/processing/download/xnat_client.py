from pathlib import Path
import configparser
import json
import netrc
import shutil
import tempfile
import zipfile
from typing import Dict, List
from urllib.parse import urlparse

from pyxnat import Interface


def _debug_log(verbose: bool, message: str, *args):
    if verbose:
        print("XNAT DEBUG:", message % args if args else message)


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


def _get_xnat_credentials(config):
    xnat_config = config["xnat"]
    server = xnat_config.get("server")
    project_id = xnat_config.get("project_id")
    if not server or not project_id:
        raise ValueError("xnat.server and xnat.project_id must be set in the pipeline config.")

    credentials_file = xnat_config.get("credentials_file")
    username = xnat_config.get("username")
    password = xnat_config.get("password")

    if credentials_file:
        creds = load_xnat_credentials(credentials_file)
        username = username or creds.get("username")
        password = password or creds.get("password")

    if not username or not password:
        creds = load_xnat_netrc_credentials(server)
        username = username or creds.get("username")
        password = password or creds.get("password")

    if not username or not password:
        raise ValueError(
            "XNAT credentials must be provided via xnat.credentials_file, xnat.username/xnat.password, or ~/.netrc."
        )

    return server, project_id, username, password, bool(xnat_config.get("verbose", False)), bool(xnat_config.get("verify_ssl", True))


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


def load_xnat_netrc_credentials(server: str) -> Dict[str, str]:
    parsed = urlparse(server)
    host = parsed.hostname or server
    try:
        auths = netrc.netrc()
    except Exception as exc:
        raise ValueError(f"Unable to read ~/.netrc: {exc}")

    if host not in auths.hosts:
        raise ValueError(f"No credentials found for machine '{host}' in ~/.netrc.")

    login, account, password = auths.authenticators(host)
    if not login or not password:
        raise ValueError(f"Incomplete credentials for machine '{host}' in ~/.netrc.")

    return {"username": login, "password": password}


def list_project_subjects(interface: Interface, project_id: str, verbose: bool = False):
    project = interface.select.project(project_id)
    if not project.exists():
        raise ValueError(f"XNAT project '{project_id}' not found.")

    subjects = project.subjects()
    labels = _selection_get(subjects)
    if labels:
        return labels

    fallback = interface.select(f"/projects/{project_id}/subjects/*")
    return _selection_get(fallback)


def list_subject_experiments(interface: Interface, project_id: str, subject_id: str, verbose: bool = False) -> List[str]:
    _debug_log(verbose, "Querying XNAT for project '%s' subject '%s'", project_id, subject_id)
    subject_obj = interface.select.project(project_id).subject(subject_id)
    exists = subject_obj.exists()
    _debug_log(verbose, "XNAT subject exists check returned: %s", exists)
    if not exists:
        raise ValueError(f"XNAT subject '{subject_id}' not found in project '{project_id}'.")

    experiments = None
    try:
        experiments = subject_obj.experiments()
    except Exception as exc:
        _debug_log(verbose, "pyxnat subject.experiments() raised: %s", exc)
        raise

    _debug_log(verbose, "pyxnat subject.experiments() raw result: %r", experiments)
    labels = _selection_get(experiments)
    _debug_log(verbose, "Parsed experiment labels from subject.experiments(): %r", labels)
    if labels:
        return labels

    fallback = interface.select(f"/projects/{project_id}/subjects/{subject_id}/experiments/*")
    _debug_log(verbose, "Using fallback selection path: /projects/%s/subjects/%s/experiments/*", project_id, subject_id)
    _debug_log(verbose, "pyxnat fallback raw result: %r", fallback)
    labels = _selection_get(fallback)
    _debug_log(verbose, "Parsed experiment labels from fallback selection: %r", labels)
    return labels


def get_xnat_experiment_label(interface: Interface, project_id: str, subject_id: str, experiment_id: str, verbose: bool = False) -> str:
    exp = interface.select.project(project_id).subject(subject_id).experiment(experiment_id)
    label = None
    try:
        if hasattr(exp, "label"):
            label = exp.label()
    except Exception as exc:
        _debug_log(verbose, "Unable to read experiment label for %s: %s", experiment_id, exc)

    if not label:
        try:
            attrs = getattr(exp, "attrs", None)
            if attrs and hasattr(attrs, "get"):
                label = attrs.get("label")
        except Exception as exc:
            _debug_log(verbose, "Unable to read experiment attrs for %s: %s", experiment_id, exc)

    return str(label) if label else str(experiment_id)


def _xnat_experiment_exists(interface: Interface, project_id: str, subject_id: str, experiment_label: str, verbose: bool = False) -> bool:
    subject_id = subject_id
    try:
        exp = interface.select.project(project_id).subject(subject_id).experiment(experiment_label)
        exists = exp.exists()
        _debug_log(verbose, "XNAT experiment exists check for %s/%s: %s", subject_id, experiment_label, exists)
        return bool(exists)
    except Exception as exc:
        _debug_log(verbose, "Unable to verify experiment %s for subject %s: %s", experiment_label, subject_id, exc)
        return False


def _download_resource_contents(resource, target_dir: Path, verbose: bool = False) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(resource, "get"):
        try:
            _debug_log(verbose, "Attempting bulk download for resource '%s' to %s", resource, target_dir)
            resource.get(str(target_dir))
            downloaded = len([p for p in target_dir.rglob("*") if p.is_file()])
            if downloaded:
                return downloaded
        except Exception as exc:
            _debug_log(verbose, "Bulk download failed for resource '%s': %s", resource, exc)

    file_names = _selection_get(resource.files())
    if not file_names:
        return 0

    downloaded = 0
    for file_name in sorted(file_names):
        target_file = target_dir / file_name
        file_obj = resource.file(file_name)
        if hasattr(file_obj, "get_copy"):
            file_obj.get_copy(str(target_file))
        else:
            file_obj.get(str(target_file))
        downloaded += 1
    return downloaded


def _cleanup_empty_dirs(path):
    for child in sorted(path.iterdir(), key=lambda p: len(p.parts), reverse=True):
        if child.is_dir():
            _cleanup_empty_dirs(child)
            try:
                child.rmdir()
            except OSError:
                pass


def _flatten_scan_contents(scan_dir: Path, verbose: bool = False):
    _debug_log(verbose, "Flattening scan folder %s", scan_dir)
    nested_files = [p for p in scan_dir.rglob("*") if p.is_file() and p.parent != scan_dir]
    for file_path in nested_files:
        target_file = scan_dir / file_path.name
        suffix = 1
        while target_file.exists():
            target_file = scan_dir / f"{file_path.stem}_{suffix}{file_path.suffix}"
            suffix += 1
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(target_file))
    _cleanup_empty_dirs(scan_dir)


def _extract_and_squash_scans_archive(archive_path: Path, destination: Path, verbose: bool = False):
    _debug_log(verbose, "Extracting scans archive %s to %s", archive_path, destination)
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(destination)

    scan_root = destination / "scans"
    if not scan_root.exists():
        subdirs = [p for p in destination.iterdir() if p.is_dir()]
        if len(subdirs) == 1 and (subdirs[0] / "scans").exists():
            scan_root = subdirs[0] / "scans"
        elif any(p.name.startswith("scans") and p.is_dir() for p in subdirs):
            scan_root = next(p for p in subdirs if p.name.startswith("scans"))
        else:
            scan_root = destination

    if scan_root.name == "scans":
        for scan_dir in sorted(scan_root.iterdir()):
            if not scan_dir.is_dir():
                continue
            target_dir = destination / scan_dir.name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(scan_dir), str(target_dir))
        shutil.rmtree(scan_root, ignore_errors=True)

    for scan_dir in sorted(destination.iterdir()):
        if not scan_dir.is_dir():
            continue
        _flatten_scan_contents(scan_dir, verbose=verbose)

    _cleanup_empty_dirs(destination)


def download_experiment(interface: Interface, project_id: str, subject_id: str, experiment_label: str, destination: Path, verbose: bool = False):
    exp = interface.select.project(project_id).subject(subject_id).experiment(experiment_label)
    destination.mkdir(parents=True, exist_ok=True)

    scans = exp.scans()
    scan_labels = _selection_get(scans)
    if scan_labels:
        _debug_log(
            verbose,
            "Downloading experiment '%s' scans %r for subject '%s'",
            experiment_label,
            scan_labels,
            subject_id,
        )
        temp_dir = Path(tempfile.mkdtemp(prefix="hbicproc_xnat_scans_"))
        try:
            archive_path = scans.download(str(temp_dir), type="ALL", extract=False)
            if isinstance(archive_path, (list, tuple)):
                archive_path = archive_path[0]
            archive_path = Path(archive_path)
            if not archive_path.exists():
                raise ValueError(f"Scan archive download failed; archive not found at {archive_path}")
            _extract_and_squash_scans_archive(archive_path, destination, verbose=verbose)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return

    resources = exp.resources()
    resource_names = _selection_get(resources)
    if not resource_names:
        raise ValueError(f"No resources found for experiment '{experiment_label}'.")

    files_downloaded = 0
    for resource_name in sorted(resource_names):
        resource = exp.resource(resource_name)
        target_dir = destination / resource_name
        files_downloaded += _download_resource_contents(resource, target_dir, verbose=verbose)

    if not files_downloaded:
        raise ValueError(f"No files were downloaded for experiment '{experiment_label}' under subject '{subject_id}'.")
