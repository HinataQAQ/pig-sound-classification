from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf

from mix_audio_at_snr import demand_environment_recording_id, file_md5, file_sha256


DEMAND_ENVIRONMENT_NOTES = {
    "DWASHING": {
        "family": "steady_indoor_mechanical",
        "description": "Domestic washroom with a front-loading washer running a wash cycle.",
    },
    "TBUS": {
        "family": "engine_or_transportation_machinery",
        "description": "Transportation recording inside a public transit bus.",
    },
    "STRAFFIC": {
        "family": "mixed_environmental_noise",
        "description": "Street recording at a busy traffic intersection.",
    },
}


LOCAL_TEMPLATE_COLUMNS = [
    "noise_file",
    "noise_type",
    "original_source_file",
    "source_session_id",
    "farm_or_location",
    "recording_date",
    "recording_device",
    "collector",
    "owner",
    "permission_or_license",
    "allowed_for_publication",
    "allowed_for_redistribution",
    "md5",
    "notes",
]


def read_zenodo_record(record_url: str) -> dict[str, Any]:
    with urllib.request.urlopen(record_url, timeout=60) as response:
        return json.load(response)


def file_info(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "duration_s": float(info.frames) / float(info.samplerate),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "md5": file_md5(path),
        "sha256": file_sha256(path),
    }


def demand_archive_record(zenodo: dict[str, Any], archive_name: str) -> dict[str, Any]:
    for item in zenodo.get("files", []):
        if item.get("key") == archive_name:
            checksum = str(item.get("checksum", ""))
            return {
                "archive_name": archive_name,
                "archive_url": item.get("links", {}).get("self"),
                "archive_size": int(item.get("size", 0)),
                "archive_md5_zenodo": checksum.replace("md5:", "") if checksum.startswith("md5:") else checksum,
            }
    raise FileNotFoundError(f"Archive not found in Zenodo record: {archive_name}")


def write_local_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LOCAL_TEMPLATE_COLUMNS)
        writer.writeheader()


def build_demand_manifest(
    *,
    demand_root: Path,
    zenodo: dict[str, Any],
    selected_environments: list[str],
    selected_channel: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    archive_records: list[dict[str, Any]] = []
    metadata = zenodo.get("metadata", {})
    license_meta = metadata.get("license", {})
    for env in selected_environments:
        env = env.strip().upper()
        archive_name = f"{env}_48k.zip"
        archive = demand_archive_record(zenodo, archive_name)
        archive_path = demand_root / "downloads" / archive_name
        if not archive_path.exists():
            raise FileNotFoundError(f"Downloaded archive is missing: {archive_path}")
        archive["archive_md5_local"] = file_md5(archive_path)
        archive["archive_sha256_local"] = file_sha256(archive_path)
        if archive["archive_md5_zenodo"] and archive["archive_md5_local"] != archive["archive_md5_zenodo"]:
            raise RuntimeError(f"Archive MD5 mismatch for {archive_name}")
        archive_records.append(archive)

        channel_file = f"ch{selected_channel:02d}.wav"
        audio_path = demand_root / "audio" / env / channel_file
        if not audio_path.exists():
            raise FileNotFoundError(f"Selected channel WAV is missing: {audio_path}")
        info = file_info(audio_path)
        note = DEMAND_ENVIRONMENT_NOTES.get(env, {})
        rows.append(
            {
                "dataset": "DEMAND",
                "noise_environment": env,
                "environment_family": note.get("family", ""),
                "description": note.get("description", ""),
                "environment_recording_id": demand_environment_recording_id(env, channel_file),
                "noise_file": audio_path.as_posix(),
                "noise_file_project_relative": audio_path.as_posix(),
                "noise_file_md5": info["md5"],
                "noise_file_sha256": info["sha256"],
                "selected_channel": selected_channel,
                "channel_file": channel_file,
                "original_sample_rate": info["sample_rate"],
                "channels_in_selected_file": info["channels"],
                "duration_s": info["duration_s"],
                "frames": info["frames"],
                "archive_name": archive_name,
                "archive_md5": archive["archive_md5_local"],
                "archive_sha256": archive["archive_sha256_local"],
                "license_id": license_meta.get("id", ""),
                "allowed_for_publication": True,
                "allowed_for_redistribution": True,
                "treat_channels_as_independent": False,
                "paper_facing_allowed": True,
            }
        )
    provenance = {
        "dataset_name": metadata.get("title", "DEMAND"),
        "doi": metadata.get("doi"),
        "zenodo_record": zenodo.get("id"),
        "zenodo_record_url": zenodo.get("links", {}).get("html") or f"https://zenodo.org/records/{zenodo.get('id')}",
        "version": metadata.get("version"),
        "download_date_utc": datetime.now(timezone.utc).date().isoformat(),
        "displayed_license_metadata": license_meta,
        "license_note": (
            "Zenodo metadata currently reports CC-BY-4.0, while DEMAND.pdf text states "
            "Creative Commons Attribution-ShareAlike 3.0 Unported. Record both and cite explicitly."
        ),
        "citation": metadata.get("creators", []),
        "selected_channel_policy": (
            f"Use channel {selected_channel} for each selected DEMAND environment; synchronized channels "
            "are not treated as independent recordings."
        ),
        "selected_environments": selected_environments,
        "archives": archive_records,
    }
    return pd.DataFrame(rows), provenance


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit DEMAND and local reviewed noise assets for simulated robustness experiments.")
    ap.add_argument("--demand_root", default="data/noise_sources/demand")
    ap.add_argument("--zenodo_record_api", default="https://zenodo.org/api/records/1227121")
    ap.add_argument("--selected_environments", nargs="+", default=["DWASHING", "TBUS", "STRAFFIC"])
    ap.add_argument("--selected_channel", type=int, default=1)
    ap.add_argument("--write_local_template", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    demand_root = Path(args.demand_root)
    demand_root.mkdir(parents=True, exist_ok=True)
    zenodo = read_zenodo_record(args.zenodo_record_api)
    manifest, provenance = build_demand_manifest(
        demand_root=demand_root,
        zenodo=zenodo,
        selected_environments=args.selected_environments,
        selected_channel=args.selected_channel,
    )
    manifest_path = demand_root / "NOISE_SOURCE_MANIFEST.csv"
    provenance_path = demand_root / "PROVENANCE.json"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.write_local_template:
        write_local_template(Path("data/noise_sources/local_reviewed/PROVENANCE_TEMPLATE.csv"))
    print(f"[OK] wrote {manifest_path}")
    print(f"[OK] wrote {provenance_path}")


if __name__ == "__main__":
    main()
