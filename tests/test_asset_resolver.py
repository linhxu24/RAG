import uuid
from types import SimpleNamespace

from app.assets.resolver import AssetResolver, detect_asset_tokens
from app.constants import Intent
from app.generation.renderer import ResponseRenderer
from app.generation.schemas import GeneratedResponse, ResultBody, ResultItem
from app.ingestion.asset_masker import append_asset_tokens, mask_asset_positions


class FakeScalarResult:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values


class FakeSession:
    def __init__(self, values):
        self.values = values

    def scalars(self, _statement):
        return FakeScalarResult(self.values)


def test_detect_and_append_asset_tokens():
    token = "[asset:doc_001_img_0001]"
    text = append_asset_tokens("Bàn chải Oral-B", [token, token])
    assert detect_asset_tokens(text) == [token]
    assert text.count(token) == 1


def test_detect_asset_tokens_ignores_template_placeholders():
    assert detect_asset_tokens(
        "Token thật [asset:test_product_01], token mẫu [asset:test_product_XX]."
    ) == ["[asset:test_product_01]"]


def test_asset_resolver_maps_token_to_url(tmp_path):
    local_file = tmp_path / "image.png"
    local_file.write_bytes(b"image")
    asset = SimpleNamespace(
        asset_id=uuid.uuid4(),
        asset_token="[asset:doc_001_img_0001]",
        public_url="/assets/doc_001/image.png",
        local_path=str(local_file),
        asset_type="product_image",
    )
    result = AssetResolver().resolve(
        FakeSession([asset]),
        f"Ảnh: {asset.asset_token} và [asset:missing]",
    )
    assert result.assets[0]["url"] == "/assets/doc_001/image.png"
    assert result.assets[0]["local_file_exists"] is True
    assert result.missing_assets == ["[asset:missing]"]


def test_asset_resolver_maps_item_asset_id_without_text_token(tmp_path):
    local_file = tmp_path / "image.png"
    local_file.write_bytes(b"image")
    asset_id = uuid.uuid4()
    asset = SimpleNamespace(
        asset_id=asset_id,
        asset_token="[asset:stable_hash]",
        stable_asset_key="stable_hash",
        public_url="/assets/doc/image.png",
        local_path=str(local_file),
        asset_type="product_image",
    )

    result = AssetResolver().resolve(
        FakeSession([asset]),
        "Tìm thấy 1 sản phẩm phù hợp.",
        asset_ids=[str(asset_id)],
    )

    assert [item["asset_id"] for item in result.assets] == [str(asset_id)]
    assert result.assets[0]["token"] == "[asset:stable_hash]"
    assert result.missing_assets == []


def test_asset_resolver_deduplicates_token_and_item_asset_id(tmp_path):
    local_file = tmp_path / "image.png"
    local_file.write_bytes(b"image")
    asset_id = uuid.uuid4()
    asset = SimpleNamespace(
        asset_id=asset_id,
        asset_token="[asset:stable_hash]",
        stable_asset_key="stable_hash",
        public_url="/assets/doc/image.png",
        local_path=str(local_file),
        asset_type="product_image",
    )

    result = AssetResolver().resolve(
        FakeSession([asset]),
        "Ảnh: [asset:stable_hash]",
        asset_ids=[str(asset_id)],
    )

    assert len(result.assets) == 1


def test_renderer_resolves_assets_from_structured_list_items(tmp_path):
    local_file = tmp_path / "image.png"
    local_file.write_bytes(b"image")
    asset_id = uuid.uuid4()
    asset = SimpleNamespace(
        asset_id=asset_id,
        asset_token="[asset:stable_hash]",
        stable_asset_key="stable_hash",
        public_url="/assets/doc/image.png",
        local_path=str(local_file),
        asset_type="product_image",
    )
    response = GeneratedResponse(
        intent=Intent.PRODUCT_LIST,
        confidence=0.97,
        answer_type="direct_data",
        result=ResultBody(
            text="Tìm thấy 1 sản phẩm phù hợp.",
            items=[
                ResultItem(
                    type="product",
                    id=str(uuid.uuid4()),
                    asset_ids=[str(asset_id)],
                )
            ],
        ),
    )

    rendered = ResponseRenderer().resolve_assets(FakeSession([asset]), response)

    assert [item["asset_id"] for item in rendered.result.assets] == [
        str(asset_id)
    ]


def test_renderer_resolves_asset_tokens_from_item_data(tmp_path):
    local_file = tmp_path / "image.png"
    local_file.write_bytes(b"image")
    asset_id = uuid.uuid4()
    asset = SimpleNamespace(
        asset_id=asset_id,
        asset_token="[asset:table_row_hash]",
        stable_asset_key="table_row_hash",
        public_url="/assets/doc/table-row.png",
        local_path=str(local_file),
        asset_type="table_image",
    )
    response = GeneratedResponse(
        intent=Intent.FAQ,
        confidence=0.9,
        answer_type="rag",
        result=ResultBody(
            text="Theo dữ liệu hiện có.",
            items=[
                ResultItem(
                    type="table_row",
                    id=str(uuid.uuid4()),
                    data={"image": "[asset:table_row_hash]"},
                )
            ],
        ),
    )

    rendered = ResponseRenderer().resolve_assets(FakeSession([asset]), response)

    assert [item["asset_id"] for item in rendered.result.assets] == [str(asset_id)]
    assert rendered.result.missing_assets == []


def test_asset_masking_replaces_docling_placeholder_in_position():
    token = "[asset:doc_001_img_0001]"
    text = mask_asset_positions("Trước ảnh\n<!-- image -->\nSau ảnh", [token])
    assert text == f"Trước ảnh\n{token}\nSau ảnh"
