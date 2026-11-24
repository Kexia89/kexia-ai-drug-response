from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests

CATALOG_URL = "https://api.cellmodelpassports.sanger.ac.uk/download_files?include=groups.files"
CATALOG_PAGE = "https://cellmodelpassports.sanger.ac.uk/downloads/gdsc"


@dataclass(frozen=True)
class Resource:
    key: str
    group_title: str
    url: str
    expected_size: int | None
    upstream_timestamp: str | None
    license_note: str
    access_note: str = "Public download; no credentials required"

    @property
    def filename(self) -> str:
        return Path(urlparse(self.url).path).name


RESOURCE_TITLES = {
    "gdsc2_fitted": "GDSC2 IC50 Data",
    "gdsc2_raw": "GDSC2 Raw Data",
    "compounds": "Compound Annotation",
    "models": "Model List",
    "mutations": "Mutations Summary",
    "cnv": "CNV Summary",
    "growth": "Growth Rate",
    "rnaseq": "RNA-Seq Summary",
}

LICENSE_NOTE = (
    "Wellcome Sanger Institute / Cell Model Passports data. Reuse remains subject "
    "to the upstream Data Usage Policy and Terms & Conditions; consult source before redistribution."
)


def _timestamp_key(value: str | None) -> str:
    return value or ""


def discover_resources(session: requests.Session | None = None) -> dict[str, Resource]:
    """Resolve current official files from the documented CMP JSON:API catalog."""
    s = session or requests.Session()
    response = s.get(CATALOG_URL, headers={"Accept": "application/vnd.api+json"}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    groups = {str(x["id"]): x for x in payload.get("included", []) if x["type"] == "download_file_group"}
    files = {str(x["id"]): x for x in payload.get("included", []) if x["type"] == "download_file"}
    by_title = {g["attributes"]["title"]: g for g in groups.values()}
    found: dict[str, Resource] = {}
    for key, title in RESOURCE_TITLES.items():
        group = by_title.get(title)
        if not group:
            continue
        file_ids = [str(x["id"]) for x in group.get("relationships", {}).get("files", {}).get("data", [])]
        candidates = [files[x] for x in file_ids if x in files]
        if not candidates:
            continue
        latest = max(candidates, key=lambda x: _timestamp_key(x["attributes"].get("timestamp")))
        attrs = latest["attributes"]
        found[key] = Resource(
            key=key,
            group_title=title,
            url=attrs["url"],
            expected_size=attrs.get("size"),
            upstream_timestamp=attrs.get("timestamp"),
            license_note=LICENSE_NOTE,
        )
    return found


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resource(resource: Resource, destination: Path, retries: int = 3) -> dict:
    """Stream to a `.part` file, resume with Range when supported, and atomically finalize."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (resource.expected_size is None or destination.stat().st_size == resource.expected_size):
        return _manifest_record(resource, destination, "reused")
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(resource.url, stream=True, timeout=(30, 180), headers=headers) as response:
                response.raise_for_status()
                append = existing > 0 and response.status_code == 206
                if existing and not append:
                    partial.unlink(missing_ok=True)
                with partial.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            handle.flush()
            if resource.expected_size is not None and partial.stat().st_size != resource.expected_size:
                raise IOError(
                    f"Size mismatch for {resource.key}: {partial.stat().st_size} != {resource.expected_size}"
                )
            os.replace(partial, destination)
            return _manifest_record(resource, destination, "downloaded")
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _manifest_record(resource: Resource, path: Path, status: str) -> dict:
    project_root = Path(__file__).resolve().parents[3]
    portable_path = path.resolve()
    local_path = (
        portable_path.relative_to(project_root).as_posix()
        if portable_path.is_relative_to(project_root)
        else portable_path.as_posix()
    )
    return {
        **asdict(resource),
        "catalog_url": CATALOG_URL,
        "catalog_page": CATALOG_PAGE,
        "local_path": local_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }


def write_manifest(records: Iterable[dict], path: Path, errors: Iterable[dict] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_resources: dict[str, dict] = {}
    existing_errors: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_resources = {str(row.get("key")): row for row in existing.get("resources", [])}
            existing_errors = list(existing.get("errors", []))
        except (json.JSONDecodeError, OSError):
            pass
    for record in records:
        existing_resources[str(record.get("key"))] = record
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "resources": list(existing_resources.values()),
        "errors": existing_errors + list(errors),
        "access_constraints": {
            "DepMap_Broad": "Some current releases may require account acceptance; not bypassed.",
            "cBioPortal": "Not needed for the first verified GDSC2 run.",
            "raw_genomics": "Not downloaded: processed matrices/driver summaries were preferred.",
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

