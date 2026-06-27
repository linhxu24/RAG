"""Smoke-check renderer to asset resolver handoff."""

import uuid

from app.assets.resolver import AssetResolution
from app.constants import Intent
from app.generation.renderer import ResponseRenderer
from app.generation.schemas import (
    GeneratedResponse,
    ResultBody,
    ResultItem,
    SourceReference,
)


class RecordingAssetResolver:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def resolve(self, session, text, asset_ids=None, doc_ids=None):
        self.calls.append(
            {
                "text": text,
                "asset_ids": list(asset_ids or []),
                "doc_ids": list(doc_ids or []),
            }
        )
        return AssetResolution(text=text, assets=[], missing_assets=[])


def main() -> None:
    doc_id = str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    renderer = ResponseRenderer()
    recorder = RecordingAssetResolver()
    renderer.asset_resolver = recorder
    response = GeneratedResponse(
        intent=Intent.PRODUCT_LIST,
        confidence=1.0,
        answer_type="direct_data",
        result=ResultBody(
            text="Ảnh: [asset:stable_hash]",
            items=[
                ResultItem(
                    type="product",
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    asset_ids=[asset_id],
                )
            ],
            sources=[
                SourceReference(
                    source_type="product",
                    source_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                )
            ],
        ),
    )
    renderer.resolve_assets(None, response)
    assert recorder.calls == [
        {
            "text": "Ảnh: [asset:stable_hash]",
            "asset_ids": [asset_id],
            "doc_ids": [doc_id],
        }
    ]
    print(recorder.calls[0])


if __name__ == "__main__":
    main()
