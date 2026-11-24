from __future__ import annotations

import json

from kexia_cancer.data.download import discover_resources, write_manifest


class Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "included": [
                {
                    "id": "46",
                    "type": "download_file_group",
                    "attributes": {"title": "GDSC2 IC50 Data"},
                    "relationships": {"files": {"data": [{"id": "old"}, {"id": "new"}]}},
                },
                {"id": "old", "type": "download_file", "attributes": {"url": "https://official/old.xlsx", "size": 1, "timestamp": "2023-01-01"}},
                {"id": "new", "type": "download_file", "attributes": {"url": "https://official/new.xlsx", "size": 2, "timestamp": "2024-01-01"}},
            ]
        }


class Session:
    def get(self, *args, **kwargs):
        return Response()


def test_catalog_discovery_selects_latest_version() -> None:
    resources = discover_resources(Session())
    assert resources["gdsc2_fitted"].url == "https://official/new.xlsx"
    assert resources["gdsc2_fitted"].expected_size == 2


def test_manifest_updates_without_losing_previous_resources(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest([{"key": "gdsc2_fitted", "sha256": "a"}], path)
    write_manifest([{"key": "rnaseq", "sha256": "b"}], path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert {row["key"] for row in payload["resources"]} == {"gdsc2_fitted", "rnaseq"}

