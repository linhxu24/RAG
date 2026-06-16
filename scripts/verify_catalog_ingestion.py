"""Verify catalog auto-detection, taxonomy, companion assets, and direct SQL retrieval."""

import csv
import tempfile
from pathlib import Path

from PIL import Image
from sqlalchemy import select

from app.admin.data_reset import delete_document
from app.config import get_settings
from app.constants import Intent
from app.db.models import Product
from app.db.session import get_session_factory
from app.ingestion.pipeline import IngestionOptions, IngestionPipeline
from app.retrieval.structured_retriever import StructuredRetriever


def main() -> None:
    settings = get_settings()
    with tempfile.TemporaryDirectory(prefix="simplydent-catalog-") as directory:
        root = Path(directory)
        image_path = root / "smoke_brush.png"
        Image.new("RGB", (8, 8), color=(20, 140, 130)).save(image_path)
        csv_path = root / "products.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "name",
                    "brand",
                    "category",
                    "description",
                    "price",
                    "currency",
                    "quantity",
                    "image_reference",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "name": "SimplyDent Smoke Brush A",
                        "brand": "SimplyDent",
                        "category": "Bàn chải điện",
                        "description": "Temporary verification record",
                        "price": "450000",
                        "currency": "VND",
                        "quantity": "2",
                        "image_reference": image_path.name,
                    },
                    {
                        "name": "SimplyDent Smoke Brush B",
                        "brand": "SimplyDent",
                        "category": "Bàn chải điện",
                        "description": "Temporary verification record",
                        "price": "650000",
                        "currency": "VND",
                        "quantity": "1",
                        "image_reference": "",
                    },
                ]
            )

        with get_session_factory()() as session:
            document, run = IngestionPipeline(settings).ingest(
                session,
                csv_path,
                IngestionOptions(
                    document_type="auto",
                    create_embeddings=False,
                    duplicate_policy="force",
                    original_file_name=csv_path.name,
                    asset_paths=(image_path,),
                ),
            )
            try:
                products = session.scalars(
                    select(Product)
                    .where(Product.source_doc_id == document.doc_id)
                    .order_by(Product.price)
                ).all()
                assert document.detected_document_type == "product_catalog"
                assert run.status == "completed"
                assert len(products) == 2
                assert all(
                    product.category_code == "ELECTRIC_TOOTHBRUSH"
                    for product in products
                )
                assert products[0].asset_id is not None

                results = StructuredRetriever().retrieve(
                    session,
                    Intent.PRODUCT_LIST,
                    "Danh sách bàn chải điện giá từ thấp đến cao",
                    [],
                )
                smoke_results = [
                    result
                    for result in results
                    if result.raw_json.get("brand") == "SimplyDent"
                ]
                assert [result.raw_json["price"] for result in smoke_results] == [
                    450000.0,
                    650000.0,
                ]
                print(
                    {
                        "doc_id": str(document.doc_id),
                        "document_type": document.detected_document_type,
                        "status": document.status,
                        "products": len(products),
                        "asset_linked": products[0].asset_id is not None,
                        "filtered_prices": [
                            result.raw_json["price"] for result in smoke_results
                        ],
                    }
                )
            finally:
                delete_document(session, settings, document.doc_id)


if __name__ == "__main__":
    main()
