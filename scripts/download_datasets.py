#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.data.download import discover_resources, download_resource, write_manifest
from kexia_cancer.paths import METADATA_DIR, RAW_DIR, ensure_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover and download official Kexia input datasets.")
    parser.add_argument(
        "--resources",
        nargs="+",
        default=["gdsc2_fitted", "gdsc2_raw", "compounds", "models", "mutations", "cnv", "growth"],
        help="Resource keys to download. Use --list to show discovered resources.",
    )
    parser.add_argument("--list", action="store_true", help="Discover and print resources without downloading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    resources = discover_resources()
    if args.list:
        for key, resource in resources.items():
            print(f"{key:15} {resource.expected_size or 0:>12} bytes  {resource.url}")
        return 0
    records, errors = [], []
    for key in args.resources:
        resource = resources.get(key)
        if resource is None:
            error = {"key": key, "error": "Not found in official API catalog"}
            errors.append(error)
            print(f"ERROR {key}: {error['error']}", file=sys.stderr)
            continue
        destination = RAW_DIR / resource.filename
        try:
            print(f"Downloading {key}: {resource.url}")
            record = download_resource(resource, destination)
            records.append(record)
            print(f"  {record['status']}: {destination.name} ({record['bytes']} bytes, sha256={record['sha256'][:12]}…)")
        except Exception as exc:  # individual failures must not abort all downloads
            errors.append({"key": key, "url": resource.url, "error": repr(exc)})
            print(f"ERROR {key}: {exc}", file=sys.stderr)
    write_manifest(records, METADATA_DIR / "download_manifest.json", errors)
    print(f"Manifest: {METADATA_DIR / 'download_manifest.json'}")
    return 1 if errors and not records else 0


if __name__ == "__main__":
    raise SystemExit(main())

