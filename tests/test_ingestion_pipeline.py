import uuid
from collections import defaultdict

import numpy as np
import pytest

from app.assets.storage import AssetStorage
from app.config import Settings
from app.ingestion.docling_parser import (
    ParsedDocument,
    ParsedTableBlock,
    ParsedTextBlock,
)
from app.ingestion.embedder import EmbeddingConfigurationError, EmbeddingService
from app.ingestion.normalizer import normalize_document
from app.ingestion.pipeline import IngestionPipeline


def test_asset_tokens_are_masked_on_matching_pages():
    blocks = [
        ParsedTextBlock("Trang một\n<!-- image -->", page_number=1),
        ParsedTextBlock("Trang hai\n<!-- image -->", page_number=2),
    ]
    IngestionPipeline._mask_assets(
        blocks,
        defaultdict(
            list,
            {
                1: ["[asset:page_1_img]"],
                2: ["[asset:page_2_img]"],
            },
        ),
    )
    assert "[asset:page_1_img]" in blocks[0].text
    assert "[asset:page_2_img]" not in blocks[0].text
    assert "[asset:page_2_img]" in blocks[1].text


def test_unplaced_asset_token_is_preserved():
    blocks = [ParsedTextBlock("Nội dung")]
    IngestionPipeline._mask_assets(
        blocks,
        defaultdict(list, {3: ["[asset:page_3_img]"]}),
    )
    assert "[asset:page_3_img]" in blocks[0].text


def test_normalizer_splits_markdown_sections_and_preserves_page_metadata():
    document = ParsedDocument(
        text_blocks=[
            ParsedTextBlock(
                "# Tổng quan\nNội dung đầu\n## Dịch vụ\nNội dung dịch vụ",
                page_number=2,
                metadata={"source_ref": "page-2"},
            )
        ]
    )
    normalized = normalize_document(document)
    assert [block.section_title for block in normalized.text_blocks] == [
        "Tổng quan",
        "Dịch vụ",
    ]
    assert all(block.page_number == 2 for block in normalized.text_blocks)
    assert normalized.metadata["normalized_counts"]["text_blocks"] == 2


def test_normalizer_unescapes_docling_asset_tokens():
    document = ParsedDocument(
        text_blocks=[ParsedTextBlock("Ảnh: [asset:test\\_product\\_01]")]
    )
    normalized = normalize_document(document)
    assert normalized.text_blocks[0].text == "Ảnh: [asset:test_product_01]"


def test_table_asset_tokens_are_prioritized_for_business_record_linking():
    parsed = ParsedDocument(
        text_blocks=[
            ParsedTextBlock(
                "Ảnh minh họa [asset:text_only_01] và [asset:example_XX]"
            )
        ],
        tables=[
            ParsedTableBlock(
                rows=[{"Tên": "Sản phẩm", "Ảnh": "[asset:product_01]"}],
                markdown="",
            )
        ],
    )
    assert IngestionPipeline._collect_authored_asset_tokens(parsed) == [
        "[asset:product_01]",
        "[asset:text_only_01]",
    ]


def test_asset_storage_uses_stable_token_and_staging(tmp_path):
    settings = Settings.model_construct(
        asset_storage_dir=tmp_path / "assets",
        asset_public_base_url="/assets",
        upload_dir=tmp_path / "uploads",
    )
    storage = AssetStorage(settings)
    first = storage.stage_bytes(
        doc_id=uuid.uuid4(),
        document_checksum="document-checksum",
        data=b"same-image",
        extension=".bin",
        index=1,
        source_key="#/pictures/0",
    )
    second = storage.stage_bytes(
        doc_id=uuid.uuid4(),
        document_checksum="document-checksum",
        data=b"same-image",
        extension=".bin",
        index=1,
        source_key="#/pictures/0",
    )
    assert first.token == second.token
    assert first.stable_asset_key == second.stable_asset_key
    assert storage.staged_file_exists(first)


class FakeEmbeddingModel:
    def __init__(self, dimension: int):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **_kwargs):
        return np.ones((len(texts), self.dimension))


def test_embedding_dimension_mismatch_fails_instead_of_padding():
    service = EmbeddingService(Settings(embedding_dim=1024))
    service._model = FakeEmbeddingModel(768)
    service._model_load_attempted = True
    with pytest.raises(EmbeddingConfigurationError, match="does not match"):
        service.embed_documents(["test"])
