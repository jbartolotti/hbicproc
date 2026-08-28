"""Exclusion-aware helpers for preprocessing.

Stub module: currently only surfaces recorded QC exclusions as a message
note. Future exclusion-driven behavior (e.g. passing excluded runs to
fMRIPrep via a BIDS filter file) belongs here.
"""

from ...core.paths import load_json


def get_exclusion_note(subject, config):
    exclusions = load_json(config["hbicproc"]["exclusions_file"], default={})
    subject_exclusions = exclusions.get(subject, {}).get("excluded_runs", [])
    if not subject_exclusions:
        return ""
    return f"Detected excluded runs: {', '.join(subject_exclusions)}. "
