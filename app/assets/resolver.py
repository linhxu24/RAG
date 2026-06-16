import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.db.models import Asset, Document

ASSET_TOKEN_PATTERN = re.compile(r"\[asset:([A-Za-z0-9_.:-]+)\]")
ASSET_TEMPLATE_SEGMENT_PATTERN = re.compile(
    r"(?:^|[_.:-])x{2,}(?:$|[_.:-])",
    flags=re.IGNORECASE,
)


@dataclass
class AssetResolution:
    text: str
    assets: list[dict[str, Any]]
    missing_assets: list[str]


def detect_asset_tokens(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            match.group(0)
            for match in ASSET_TOKEN_PATTERN.finditer(text or "")
            if not ASSET_TEMPLATE_SEGMENT_PATTERN.search(match.group(1))
        )
    )


class AssetResolver:
    def resolve(
        self,
        session: Session,
        text: str,
        asset_ids: list[str] | None = None,
    ) -> AssetResolution:
        tokens = detect_asset_tokens(text)
        requested_ids: list[uuid.UUID] = []
        invalid_ids: list[str] = []
        for value in dict.fromkeys(asset_ids or []):
            try:
                requested_ids.append(uuid.UUID(str(value)))
            except (TypeError, ValueError):
                invalid_ids.append(str(value))
        if not tokens and not requested_ids:
            return AssetResolution(
                text=text,
                assets=[],
                missing_assets=invalid_ids,
            )
        conditions = []
        if tokens:
            conditions.append(Asset.asset_token.in_(tokens))
        if requested_ids:
            conditions.append(Asset.asset_id.in_(requested_ids))
        records = session.scalars(
            select(Asset)
            .join(Document, Document.doc_id == Asset.doc_id)
            .where(
                or_(*conditions),
                Asset.status == "active",
                Document.status == "active",
            )
            .order_by(Document.uploaded_at.desc())
        ).all()
        by_token: dict[str, Asset] = {}
        by_id: dict[uuid.UUID, Asset] = {}
        for asset in records:
            by_token.setdefault(asset.asset_token, asset)
            by_id.setdefault(asset.asset_id, asset)
        assets: list[dict[str, Any]] = []
        missing: list[str] = list(invalid_ids)
        resolved_ids: set[uuid.UUID] = set()
        for token in tokens:
            asset = by_token.get(token)
            if asset is None:
                missing.append(token)
                continue
            if asset.asset_id not in resolved_ids:
                assets.append(self._asset_payload(asset))
                resolved_ids.add(asset.asset_id)
        for asset_id in requested_ids:
            asset = by_id.get(asset_id)
            if asset is None:
                missing.append(str(asset_id))
                continue
            if asset.asset_id not in resolved_ids:
                assets.append(self._asset_payload(asset))
                resolved_ids.add(asset.asset_id)
        return AssetResolution(text=text, assets=assets, missing_assets=missing)

    @staticmethod
    def _asset_payload(asset: Asset) -> dict[str, Any]:
        return {
            "asset_id": str(asset.asset_id),
            "stable_asset_key": getattr(asset, "stable_asset_key", None),
            "token": asset.asset_token,
            "url": asset.public_url or asset.local_path,
            "type": asset.asset_type or "image",
            "local_file_exists": bool(
                asset.local_path and Path(asset.local_path).is_file()
            ),
        }
