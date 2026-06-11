from pathlib import Path
import csv
import configparser
import json
import netrc
import re
import shutil
import sys
import tempfile
import time
import zipfile
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pyxnat import Interface
from ..state import load_subject_state, save_subject_state, update_session_state


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


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _raw_session_name(session_value: str, subject_id: str) -> str:
    return strip_session_prefix(normalize_session_label(session_value, subject_id))


def _disk_sessions(output_root: Path) -> set[tuple[str, str]]:
    sessions = set()
    if not output_root.exists() or not output_root.is_dir():
        return sessions
    for subject_dir in sorted(output_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = normalize_subject_label(subject_dir.name)
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session_name = strip_session_prefix(session_dir.name)
            if any(session_dir.rglob("*")):
                sessions.add((subject_id, session_name))
    return sessions


def _normalize_xnat_subject_id(subject_label: str) -> str:
    if not subject_label:
        return ""
    if "/" in subject_label:
        subject_label = subject_label.split("/", 1)[0]
    return normalize_subject_label(subject_label)


def _normalize_xnat_experiment_label(experiment_label: str) -> str:
    if not experiment_label:
        return ""
    if "/" in experiment_label:
        experiment_label = experiment_label.split("/", 1)[-1]
    return experiment_label.strip()


def _clean_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower()) if label else ""


def _find_xnat_subject_and_experiment_for_session_sic(interface: Interface, project_id: str, session_sic: str, verbose: bool = False) -> Optional[Tuple[str, str]]:
    session_sic = _normalize_xnat_experiment_label(session_sic)
    cleaned_target = _clean_label(session_sic)
    subjects = list_project_subjects(interface, project_id, verbose=verbose)
    for raw_subject in sorted(subjects, key=lambda s: str(s)):
        try:
            experiment_ids = list_subject_experiments(interface, project_id, raw_subject, verbose=verbose)
        except Exception:
            continue

        exact_matches = [exp_id for exp_id in experiment_ids if exp_id == session_sic]
        if len(exact_matches) == 1:
            return raw_subject, exact_matches[0]

        fuzzy_matches = [exp_id for exp_id in experiment_ids if _clean_label(exp_id) == cleaned_target]
        if len(fuzzy_matches) == 1:
            return raw_subject, fuzzy_matches[0]

        for exp_id in experiment_ids:
            try:
                actual_label = get_xnat_experiment_label(interface, project_id, raw_subject, exp_id, verbose=verbose)
            except Exception:
                continue
            if actual_label == session_sic or _clean_label(actual_label) == cleaned_target:
                return raw_subject, exp_id

    return None


def _xnat_experiment_exists(interface: Interface, project_id: str, subject_id: str, experiment_label: str, verbose: bool = False) -> bool:
    subject_id = _normalize_xnat_subject_id(subject_id)
    experiment_label = _normalize_xnat_experiment_label(experiment_label)
    try:
        exp = interface.select.project(project_id).subject(subject_id).experiment(experiment_label)
        exists = exp.exists()
        _debug_log(verbose, "XNAT experiment exists check for %s/%s: %s", subject_id, experiment_label, exists)
        return bool(exists)
    except Exception as exc:
        _debug_log(verbose, "Unable to verify experiment %s for subject %s: %s", experiment_label, subject_id, exc)
        return False


def download_all(config, dry_run=False, rerun=False):
    xnat_config = config["xnat"]
    session_names_file = xnat_config.get("session_names_file")
    if not session_names_file:
        raise ValueError("xnat.session_names_file is required for --all.")

    delimiter = xnat_config.get("session_names_delimiter")
    session_map = load_session_map(session_names_file, delimiter=delimiter)
    output_root = Path(xnat_config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    existing_sessions = _disk_sessions(output_root)
    plan = []
    skipped = 0

    for xnat_label, entry in sorted(session_map.items()):
        subject_value = entry.get("subject")
        if not subject_value:
            print(f"WARNING: session_names entry '{xnat_label}' has no subject; skipping.")
            skipped += 1
            continue

        subject_id = _normalize_xnat_subject_id(subject_value)
        session_name = _raw_session_name(entry.get("session_value"), subject_id)
        if not session_name:
            print(f"WARNING: session_names entry '{xnat_label}' has invalid session value; skipping.")
            skipped += 1
            continue

        if not rerun and (subject_id, session_name) in existing_sessions:
            skipped += 1
            continue

        plan.append({
            "session_sic": _normalize_xnat_experiment_label(xnat_label),
            "output_subject": subject_id,
            "session_name": session_name,
        })

    if not plan:
        print("No sessions need downloading from session_names.tsv.")
        return 0

    print(f"Found {len(session_map)} session mappings in {session_names_file}.")
    print(f"{len(existing_sessions)} sessions are already on disk.")
    print(f"Preparing to download {len(plan)} sessions:")
    for entry in plan:
        print(f"  {entry['output_subject']} {entry['session_name']}")

    if dry_run:
        print("Dry run: no downloads will be performed.")
        return 0

    server, project_id, username, password, verbose, verify_ssl = _get_xnat_credentials(config)
    interface = Interface(server=server, user=username, password=password, verify=verify_ssl)

    available = []
    missing = []
    for entry in plan:
        output_subject = entry["output_subject"]
        session_sic = _normalize_xnat_experiment_label(entry["session_sic"])
        xnat_subject, experiment_id = _find_xnat_subject_and_experiment_for_session_sic(
            interface,
            project_id,
            session_sic,
            verbose=verbose,
        )
        if experiment_id and xnat_subject:
            available.append({
                **entry,
                "xnat_subject": xnat_subject,
                "session_sic": session_sic,
                "experiment_id": experiment_id,
            })
        else:
            missing.append({
                **entry,
                "xnat_subject": None,
                "session_sic": session_sic,
            })

    for entry in missing:
        print(
            f"WARNING: XNAT experiment not found for output subject {entry['output_subject']} session {entry['session_name']} "
            f"({entry['session_sic']})."
        )

    if not available:
        print("No available XNAT sessions to download.")
        return 1

    total = len(available)
    completed = 0
    elapsed_total = 0.0
    for index, entry in enumerate(available, start=1):
        output_subject = entry["output_subject"]
        xnat_subject = entry["xnat_subject"]
        session_name = entry["session_name"]
        session_sic = entry["session_sic"]
        experiment_id = entry["experiment_id"]
        print(f"download {index}/{total}, {output_subject} {session_name}")

        destination = output_root / output_subject / session_name
        if destination.exists() and rerun:
            shutil.rmtree(destination)

        start = time.perf_counter()
        try:
            download_experiment(interface, project_id, xnat_subject, experiment_id, destination, verbose=verbose)
        except Exception as exc:
            print(f"ERROR: Download failed for {output_subject} {session_name}: {exc}", file=sys.stderr)
            return 1
        duration = time.perf_counter() - start
        elapsed_total += duration
        completed += 1

        update_session_state(config, f"sub-{output_subject}", session_name, {"downloaded": True, "xnat_label": session_sic})

        remaining = total - completed
        if remaining:
            average = elapsed_total / completed
            eta = average * remaining
            print(f"Estimated time remaining: {_format_duration(eta)}")
        else:
            print("Download complete.")

    _update_subject_downloaded_states(config, session_map, output_root)

    return 0


def _update_subject_downloaded_states(config, session_map, output_root: Path):
    downloaded_sessions = _disk_sessions(output_root)
    subject_sessions = {}
    for xnat_label, entry in session_map.items():
        subject_value = entry.get("subject")
        if not subject_value:
            continue
        subject_id = normalize_subject_label(subject_value)
        session_name = _raw_session_name(entry.get("session_value"), subject_id)
        if not session_name:
            continue
        subject_sessions.setdefault(subject_id, []).append(session_name)

    for subject_id, sessions in subject_sessions.items():
        expected = set(sessions)
        actual = {session for (subject, session) in downloaded_sessions if subject == subject_id}
        if expected and expected.issubset(actual):
            subject_label = f"sub-{subject_id}"
            state = load_subject_state(Path(config["study_root"]) / subject_label, config=config)
            state["downloaded"] = True
            save_subject_state(Path(config["study_root"]) / subject_label, state, config=config)
    return


def summarize_downloads(config):
    server, project_id, username, password, verbose, verify_ssl = _get_xnat_credentials(config)
    session_names_file = config["xnat"].get("session_names_file")
    delimiter = config["xnat"].get("session_names_delimiter")
    session_map = load_session_map(session_names_file, delimiter=delimiter) if session_names_file else {}
    output_root = Path(config["xnat"]["output_dir"])

    disk_sessions = set()
    xnat_records = []
    session_name_entries = {}

    interface = Interface(server=server, user=username, password=password, verify=verify_ssl)
    try:
        subjects = list_project_subjects(interface, project_id, verbose=verbose)
        for raw_subject in sorted(subjects, key=lambda s: normalize_subject_label(str(s))):
            subject_id = normalize_subject_label(str(raw_subject))
            if not subject_id:
                continue
            try:
                experiment_ids = list_subject_experiments(interface, project_id, subject_id, verbose=verbose)
            except Exception as exc:
                print(f"WARNING: Skipping XNAT subject '{raw_subject}': {exc}", file=sys.stderr)
                continue

            for exp_id in sorted(experiment_ids):
                xnat_session_name = get_xnat_experiment_label(interface, project_id, subject_id, exp_id, verbose=verbose)
                mapped = session_map.get(xnat_session_name)
                if mapped:
                    mapped_subject = normalize_subject_label(mapped.get("subject") or subject_id)
                    try:
                        session_label = normalize_session_label(mapped.get("session_value"), mapped_subject)
                    except Exception:
                        session_label = normalize_session_label(xnat_session_name, mapped_subject)
                    in_session_names = True
                else:
                    mapped_subject = subject_id
                    try:
                        session_label = normalize_session_label(xnat_session_name, mapped_subject)
                    except Exception:
                        session_label = xnat_session_name
                    in_session_names = False

                subject_value = mapped_subject
                session_value = strip_session_prefix(session_label)
                xnat_records.append({
                    "subject": subject_value,
                    "session": session_value,
                    "xnat_session": xnat_session_name,
                    "in_xnat": True,
                    "in_session_names": in_session_names,
                })

        for xnat_label, entry in session_map.items():
            subject_value = entry.get("subject")
            if not subject_value:
                continue
            mapped_subject = normalize_subject_label(subject_value)
            try:
                session_label = normalize_session_label(entry.get("session_value"), mapped_subject)
            except Exception:
                continue
            session_value = strip_session_prefix(session_label)
            key = (mapped_subject, session_value, xnat_label)
            session_name_entries[key] = {
                "subject": mapped_subject,
                "session": session_value,
                "xnat_session": xnat_label,
                "in_xnat": False,
                "in_session_names": True,
            }

        if output_root.exists() and output_root.is_dir():
            for subject_dir in sorted(output_root.iterdir()):
                if not subject_dir.is_dir():
                    continue
                subject_id = normalize_subject_label(subject_dir.name)
                for session_dir in sorted(subject_dir.iterdir()):
                    if not session_dir.is_dir():
                        continue
                    session_label = strip_session_prefix(session_dir.name)
                    disk_sessions.add((subject_id, session_label))
    finally:
        try:
            interface.disconnect()
        except Exception:
            pass

    rows = []
    seen = set()
    for record in xnat_records:
        disk_key = (record["subject"], record["session"])
        rows.append([
            record["subject"],
            record["session"],
            record["xnat_session"],
            "Yes" if record["in_xnat"] else "No",
            "Yes" if record["in_session_names"] else "No",
            "Yes" if disk_key in disk_sessions else "No",
        ])
        seen.add((record["subject"], record["session"], record["xnat_session"]))

    for key, record in session_name_entries.items():
        if key in seen:
            continue
        disk_key = (record["subject"], record["session"])
        rows.append([
            record["subject"],
            record["session"],
            record["xnat_session"],
            "Yes" if record["in_xnat"] else "No",
            "Yes" if record["in_session_names"] else "No",
            "Yes" if disk_key in disk_sessions else "No",
        ])

    for subject_id, session_label in sorted(disk_sessions):
        if not any(r[0] == subject_id and r[1] == session_label for r in rows):
            rows.append([
                subject_id,
                session_label,
                "",
                "No",
                "No",
                "Yes",
            ])

    if not rows:
        print("No download summary entries found for the configured XNAT project.")
        return 0

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    header = ["Subject", "Session", "XNAT Session", "In XNAT", "In session_names.tsv", "On disk"]
    widths = [max(len(str(cell)) for cell in column) for column in zip(header, *rows)]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))

    return 0


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


def _derive_session_label_for_xnat_experiment(subject_id: str, exp_label: str, session_map: Dict[str, Dict[str, str]]):
    if exp_label in session_map:
        entry = session_map[exp_label]
        entry_subject = entry.get("subject")
        if entry_subject and normalize_subject_label(entry_subject) != subject_id:
            return normalize_session_label(exp_label, subject_id), False
        return normalize_session_label(entry.get("session_value") or exp_label, subject_id), True
    return normalize_session_label(exp_label, subject_id), False


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
        creds = load_xnat_netrc_credentials(server)
        username = username or creds.get("username")
        password = password or creds.get("password")

    if not username or not password:
        raise ValueError(
            "XNAT credentials must be provided via xnat.credentials_file, xnat.username/xnat.password, or ~/.netrc."
        )

    session_map = {}
    session_names_file = xnat_config.get("session_names_file")
    delimiter = xnat_config.get("session_names_delimiter")
    verbose = bool(xnat_config.get("verbose", False))

    if session_names_file:
        session_map = load_session_map(session_names_file, delimiter=delimiter)

    subject_id = normalize_subject_label(subject)
    mapped_xnat_subjects = find_xnat_subjects_for_requested_subject(subject_id, session_map)

    _debug_log(verbose, "Requested subject: %s", subject)
    _debug_log(verbose, "Normalized subject for XNAT lookup: %s", subject_id)
    _debug_log(verbose, "XNAT server=%s project=%s verify_ssl=%s", server, project_id, verify_ssl)
    if session_names_file:
        _debug_log(verbose, "Loaded session names file: %s", session_names_file)
        _debug_log(verbose, "Session map entries: %d", len(session_map))
        matching_entries = {
            xnat_label: entry
            for xnat_label, entry in session_map.items()
            if entry.get("subject") and normalize_subject_label(entry.get("subject")) == subject_id
        }
        _debug_log(verbose, "Session map matches for requested subject %s: %r", subject_id, matching_entries)
    else:
        _debug_log(verbose, "No session_names_file configured; falling back to direct XNAT subject lookup for %s", subject_id)

    _debug_log(verbose, "Mapped XNAT labels for requested subject: %r", list(mapped_xnat_subjects.keys()))

    temp_root = Path(tempfile.mkdtemp(prefix=f"hbicproc_xnat_{subject_id}_"))
    try:
        interface = Interface(server=server, user=username, password=password, verify=verify_ssl)
        try:
            if mapped_xnat_subjects:
                download_plan = build_download_plan_for_mapped_subjects(
                    subject_id,
                    mapped_xnat_subjects,
                    interface,
                    project_id,
                    verbose=verbose,
                )
            else:
                _debug_log(verbose, "Using direct XNAT lookup for subject: %s", subject_id)
                experiments = list_subject_experiments(interface, project_id, subject_id, verbose=verbose)
                if not experiments:
                    raise ValueError(f"No XNAT sessions found for subject '{subject}'.")
                download_plan = build_download_plan(subject_id, experiments, session_map, verbose=verbose)

            if not download_plan:
                raise ValueError(
                    f"Unable to derive any session labels for {subject}." 
                    " Check xnat.session_names_file or session naming conventions."
                )

            subject_temp_dir = temp_root / subject
            subject_temp_dir.mkdir(parents=True, exist_ok=True)

            for (xnat_subject_id, exp_label), bids_session in download_plan.items():
                session_name = strip_session_prefix(bids_session)
                session_temp_dir = subject_temp_dir / session_name
                download_experiment(interface, project_id, xnat_subject_id, exp_label, session_temp_dir, verbose=verbose)
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


def strip_session_prefix(session_label: str) -> str:
    session_label = session_label.strip()
    if session_label.lower().startswith("ses-"):
        return session_label[4:]
    return session_label


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


def _debug_log(verbose: bool, message: str, *args):
    if verbose:
        print("XNAT DEBUG:", message % args if args else message)


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


def find_xnat_subjects_for_requested_subject(subject_id: str, session_map: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {
        xnat_label: entry
        for xnat_label, entry in session_map.items()
        if entry.get("subject") and normalize_subject_label(entry.get("subject")) == subject_id
    }


def build_download_plan_for_mapped_subjects(
    subject_id: str,
    mapped_xnat_subjects: Dict[str, Dict[str, str]],
    interface: Interface,
    project_id: str,
    verbose: bool = False,
) -> Dict[Tuple[str, str], str]:
    plan: Dict[Tuple[str, str], str] = {}
    session_labels: Dict[str, str] = {}

    for xnat_label, entry in sorted(mapped_xnat_subjects.items()):
        xnat_subject_id = xnat_label
        _debug_log(verbose, "Checking mapped XNAT label '%s' for requested subject '%s'", xnat_label, subject_id)
        _debug_log(verbose, "Mapped entry values: %r", entry)
        _debug_log(verbose, "Using XNAT subject label '%s' for query", xnat_subject_id)

        experiments = list_subject_experiments(interface, project_id, xnat_subject_id, verbose=verbose)
        if not experiments:
            raise ValueError(
                f"No XNAT sessions found for subject '{xnat_subject_id}' "
                f"for requested subject '{subject_id}'."
            )

        _debug_log(verbose, "Experiments found for XNAT subject '%s': %r", xnat_subject_id, experiments)

        session_value = entry.get("session_value") or xnat_label
        session_label = normalize_session_label(session_value, subject_id)
        _debug_log(verbose, "Derived BIDS session label '%s' from session_value '%s'", session_label, session_value)

        if session_label in session_labels and session_labels[session_label] != xnat_subject_id:
            raise ValueError(
                f"Duplicate BIDS session label '{session_label}' derived for "
                f"XNAT experiments under subject '{session_labels[session_label]}' and '{xnat_subject_id}'."
            )
        session_labels[session_label] = xnat_subject_id

        for exp_label in sorted(experiments):
            _debug_log(verbose, "Adding experiment '%s' under XNAT subject '%s' to download plan", exp_label, xnat_subject_id)
            plan[(xnat_subject_id, exp_label)] = session_label

    _debug_log(verbose, "Final download plan for mapped subjects: %r", plan)
    return plan


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


def build_download_plan(
    subject_id: str,
    experiment_labels: List[str],
    session_map: Dict[str, Dict[str, str]],
    verbose: bool = False,
) -> Dict[Tuple[str, str], str]:
    _debug_log(verbose, "Building download plan for direct subject '%s' with experiments: %r", subject_id, experiment_labels)
    plan: Dict[Tuple[str, str], str] = {}
    sessions_seen = set()
    for exp_label in sorted(experiment_labels):
        if exp_label in session_map:
            entry = session_map[exp_label]
            _debug_log(verbose, "Experiment '%s' found in session map entry: %r", exp_label, entry)
            if entry["subject"] and entry["subject"] != subject_id:
                _debug_log(verbose, "Skipping experiment '%s' because session map subject '%s' does not match requested subject '%s'", exp_label, entry["subject"], subject_id)
                continue
            session_label = normalize_session_label(entry["session_value"], subject_id)
        else:
            _debug_log(verbose, "Experiment '%s' not in session map; normalizing from XNAT label", exp_label)
            session_label = normalize_session_label(exp_label, subject_id)

        _debug_log(verbose, "Derived BIDS session label '%s' for XNAT experiment '%s'", session_label, exp_label)
        if session_label in sessions_seen:
            raise ValueError(
                f"Duplicate BIDS session label '{session_label}' derived from XNAT labels: {exp_label}."
            )
        sessions_seen.add(session_label)
        plan[(subject_id, exp_label)] = session_label

    _debug_log(verbose, "Final download plan for direct subject: %r", plan)
    return plan


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
