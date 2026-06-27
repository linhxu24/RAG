from dataclasses import fields
from inspect import signature
from typing import Any, get_type_hints
from uuid import UUID

from sqlalchemy.orm import Session

from app.assets.resolver import AssetResolution, AssetResolver
from app.assets.storage import AssetStorage, StoredAsset


def test_asset_storage_write_contract_for_ingestion():
    hints = get_type_hints(StoredAsset)
    assert [field.name for field in fields(StoredAsset)] == [
        "asset_id",
        "stable_asset_key",
        "token",
        "staged_path",
        "local_path",
        "public_url",
        "checksum",
    ]
    assert hints == {
        "asset_id": UUID,
        "stable_asset_key": str,
        "token": str,
        "staged_path": str,
        "local_path": str,
        "public_url": str,
        "checksum": str,
    }
    params = signature(AssetStorage.stage_bytes).parameters
    assert list(params) == [
        "self",
        "doc_id",
        "document_checksum",
        "data",
        "extension",
        "index",
        "source_key",
        "token_override",
    ]


def test_asset_resolution_contract_for_response_time_consumers():
    assert get_type_hints(AssetResolution) == {
        "text": str,
        "assets": list[dict[str, Any]],
        "missing_assets": list[str],
    }
    hints = get_type_hints(AssetResolver.resolve)
    assert hints["session"] is Session
    assert hints["text"] is str
    assert hints["asset_ids"] == list[str] | None
    assert hints["doc_ids"] == list[str] | None
    assert hints["return"] is AssetResolution
