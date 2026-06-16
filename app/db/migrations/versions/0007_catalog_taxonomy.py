"""Add business taxonomies, aliases, document classification and FAQ lineage."""

from alembic import op

revision = "0007_catalog_taxonomy"
down_revision = "0006_business_record_dedup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE documents
          ADD COLUMN detected_document_type TEXT,
          ADD COLUMN document_type_confidence DOUBLE PRECISION;

        CREATE TABLE product_categories (
          code TEXT PRIMARY KEY,
          display_name TEXT NOT NULL,
          parent_code TEXT REFERENCES product_categories(code) ON DELETE SET NULL,
          aliases TEXT[] NOT NULL DEFAULT '{}',
          status TEXT NOT NULL DEFAULT 'active',
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE TABLE service_categories (
          code TEXT PRIMARY KEY,
          display_name TEXT NOT NULL,
          parent_code TEXT REFERENCES service_categories(code) ON DELETE SET NULL,
          aliases TEXT[] NOT NULL DEFAULT '{}',
          status TEXT NOT NULL DEFAULT 'active',
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE TABLE faq_categories (
          code TEXT PRIMARY KEY,
          display_name TEXT NOT NULL,
          parent_code TEXT REFERENCES faq_categories(code) ON DELETE SET NULL,
          aliases TEXT[] NOT NULL DEFAULT '{}',
          status TEXT NOT NULL DEFAULT 'active',
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE TABLE category_aliases (
          alias_id BIGSERIAL PRIMARY KEY,
          entity_type TEXT NOT NULL,
          category_code TEXT NOT NULL,
          alias TEXT NOT NULL,
          normalized_alias TEXT NOT NULL,
          CONSTRAINT uq_category_alias_entity_normalized
            UNIQUE(entity_type, normalized_alias)
        );
        CREATE INDEX ix_category_aliases_entity_code
          ON category_aliases(entity_type, category_code);

        ALTER TABLE products
          ADD COLUMN brand TEXT,
          ADD COLUMN model TEXT,
          ADD COLUMN category_code TEXT REFERENCES product_categories(code) ON DELETE SET NULL,
          ADD COLUMN currency TEXT NOT NULL DEFAULT 'VND',
          ADD COLUMN source_category TEXT,
          ADD COLUMN image_reference TEXT;
        CREATE INDEX ix_products_active_category_price
          ON products(category_code, price, name)
          WHERE status = 'active';

        ALTER TABLE services
          ADD COLUMN category_code TEXT REFERENCES service_categories(code) ON DELETE SET NULL,
          ADD COLUMN currency TEXT NOT NULL DEFAULT 'VND',
          ADD COLUMN source_category TEXT,
          ADD COLUMN image_reference TEXT,
          ADD COLUMN indications TEXT[],
          ADD COLUMN contraindications TEXT[];
        CREATE INDEX ix_services_active_category_price
          ON services(category_code, price, name)
          WHERE status = 'active';

        ALTER TABLE faqs
          ADD COLUMN category_code TEXT REFERENCES faq_categories(code) ON DELETE SET NULL,
          ADD COLUMN keywords TEXT[],
          ADD COLUMN source_doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
          ADD COLUMN source_row_id UUID REFERENCES table_rows(row_id) ON DELETE SET NULL;
        UPDATE faqs
        SET source_doc_id = CASE
          WHEN metadata ->> 'source_doc_id' ~
            '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
          THEN (metadata ->> 'source_doc_id')::uuid
          ELSE NULL
        END,
        source_row_id = CASE
          WHEN metadata ->> 'source_row_id' ~
            '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
          THEN (metadata ->> 'source_row_id')::uuid
          ELSE NULL
        END;
        CREATE INDEX ix_faqs_source_doc_id ON faqs(source_doc_id);
        CREATE INDEX ix_faqs_active_category ON faqs(category_code) WHERE is_active = true;

        CREATE TABLE product_aliases (
          alias_id BIGSERIAL PRIMARY KEY,
          product_id UUID NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
          alias TEXT NOT NULL,
          normalized_alias TEXT NOT NULL,
          CONSTRAINT uq_product_alias_product_normalized
            UNIQUE(product_id, normalized_alias)
        );
        CREATE INDEX ix_product_aliases_normalized ON product_aliases(normalized_alias);

        CREATE TABLE faq_aliases (
          alias_id BIGSERIAL PRIMARY KEY,
          faq_id UUID NOT NULL REFERENCES faqs(faq_id) ON DELETE CASCADE,
          question_variant TEXT NOT NULL,
          normalized_variant TEXT NOT NULL,
          CONSTRAINT uq_faq_alias_faq_normalized
            UNIQUE(faq_id, normalized_variant)
        );
        CREATE INDEX ix_faq_aliases_normalized ON faq_aliases(normalized_variant);
        """
    )
    _seed_categories()


def _seed_categories() -> None:
    op.execute(
        """
        INSERT INTO product_categories(code, display_name, parent_code, aliases) VALUES
          ('TOOTHBRUSH', 'Bàn chải', NULL, ARRAY['bàn chải', 'toothbrush']),
          ('MANUAL_TOOTHBRUSH', 'Bàn chải thường', 'TOOTHBRUSH',
             ARRAY['bàn chải thường', 'bàn chải đánh răng', 'manual toothbrush']),
          ('ELECTRIC_TOOTHBRUSH', 'Bàn chải điện', 'TOOTHBRUSH',
             ARRAY['bàn chải điện', 'bàn chải sonic', 'electric toothbrush']),
          ('INTERDENTAL_BRUSH', 'Bàn chải kẽ răng', 'TOOTHBRUSH',
             ARRAY['bàn chải kẽ', 'bàn chải kẽ răng', 'interdental brush']),
          ('TOOTHPASTE', 'Kem đánh răng', NULL,
             ARRAY['kem đánh răng', 'toothpaste', 'kem cho răng nhạy cảm']),
          ('WATER_FLOSSER', 'Máy tăm nước', NULL,
             ARRAY['tăm nước', 'máy tăm nước', 'water flosser']),
          ('MOUTHWASH', 'Nước súc miệng', NULL, ARRAY['nước súc miệng', 'mouthwash']),
          ('DENTAL_FLOSS', 'Chỉ nha khoa', NULL, ARRAY['chỉ nha khoa', 'dental floss']),
          ('TONGUE_CLEANER', 'Dụng cụ vệ sinh lưỡi', NULL,
             ARRAY['vệ sinh lưỡi', 'cạo lưỡi', 'tongue cleaner']),
          ('WHITENING_PRODUCT', 'Sản phẩm làm trắng răng', NULL,
             ARRAY['miếng dán làm trắng răng', 'làm trắng răng', 'whitening strip']),
          ('ORTHODONTIC_CARE', 'Chăm sóc chỉnh nha', NULL,
             ARRAY['sáp chỉnh nha', 'chăm sóc niềng răng', 'orthodontic care']),
          ('NIGHT_GUARD', 'Máng bảo vệ răng', NULL,
             ARRAY['máng bảo vệ răng', 'night guard']);

        INSERT INTO service_categories(code, display_name, aliases) VALUES
          ('PREVENTIVE', 'Nha khoa dự phòng', ARRAY['dự phòng', 'vệ sinh răng', 'preventive']),
          ('RESTORATIVE', 'Phục hồi răng', ARRAY['phục hồi', 'trám răng', 'restorative']),
          ('ENDODONTIC', 'Điều trị nội nha', ARRAY['nội nha', 'điều trị tủy', 'endodontic']),
          ('PERIODONTAL', 'Điều trị nha chu', ARRAY['nha chu', 'điều trị nướu', 'periodontal']),
          ('ORTHODONTIC', 'Chỉnh nha', ARRAY['chỉnh nha', 'niềng răng', 'orthodontic']),
          ('IMPLANT', 'Cấy ghép Implant', ARRAY['implant', 'cấy ghép', 'trồng răng']),
          ('ORAL_SURGERY', 'Phẫu thuật răng miệng', ARRAY['nhổ răng', 'tiểu phẫu', 'oral surgery']),
          ('COSMETIC', 'Nha khoa thẩm mỹ', ARRAY['thẩm mỹ', 'tẩy trắng', 'cosmetic']),
          ('PEDIATRIC', 'Nha khoa trẻ em', ARRAY['trẻ em', 'nha khoa trẻ em', 'pediatric']),
          ('EMERGENCY', 'Nha khoa khẩn cấp', ARRAY['khẩn cấp', 'cấp cứu nha khoa', 'emergency']);

        INSERT INTO faq_categories(code, display_name, aliases) VALUES
          ('EMERGENCY', 'Tình huống khẩn cấp', ARRAY['khẩn cấp', 'đau dữ dội', 'sưng mặt']),
          ('POST_TREATMENT', 'Sau điều trị', ARRAY['sau điều trị', 'sau nhổ răng', 'kiêng']),
          ('ORAL_CARE', 'Chăm sóc răng miệng', ARRAY['chăm sóc răng', 'vệ sinh răng miệng']),
          ('SERVICE_INFORMATION', 'Thông tin dịch vụ', ARRAY['dịch vụ', 'quy trình', 'thời gian']),
          ('PRODUCT_USAGE', 'Sử dụng sản phẩm', ARRAY['cách dùng', 'sản phẩm']),
          ('CLINIC_POLICY', 'Chính sách phòng khám', ARRAY['chính sách', 'đặt lịch', 'thanh toán']);
        """
    )
    for entity_type, table in (
        ("product", "product_categories"),
        ("service", "service_categories"),
        ("faq", "faq_categories"),
    ):
        op.execute(
            f"""
            INSERT INTO category_aliases(entity_type, category_code, alias, normalized_alias)
            SELECT '{entity_type}', code, alias,
                   regexp_replace(
                     lower(simplydent_unaccent(alias)),
                     '[^a-z0-9]+', ' ', 'g'
                   )
            FROM {table}, unnest(aliases || ARRAY[code, display_name]) AS alias
            ON CONFLICT(entity_type, normalized_alias) DO NOTHING
            """
        )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS faq_aliases;
        DROP TABLE IF EXISTS product_aliases;
        ALTER TABLE faqs
          DROP COLUMN IF EXISTS source_row_id,
          DROP COLUMN IF EXISTS source_doc_id,
          DROP COLUMN IF EXISTS keywords,
          DROP COLUMN IF EXISTS category_code;
        ALTER TABLE services
          DROP COLUMN IF EXISTS contraindications,
          DROP COLUMN IF EXISTS indications,
          DROP COLUMN IF EXISTS image_reference,
          DROP COLUMN IF EXISTS source_category,
          DROP COLUMN IF EXISTS currency,
          DROP COLUMN IF EXISTS category_code;
        ALTER TABLE products
          DROP COLUMN IF EXISTS image_reference,
          DROP COLUMN IF EXISTS source_category,
          DROP COLUMN IF EXISTS currency,
          DROP COLUMN IF EXISTS category_code,
          DROP COLUMN IF EXISTS model,
          DROP COLUMN IF EXISTS brand;
        DROP TABLE IF EXISTS category_aliases;
        DROP TABLE IF EXISTS faq_categories;
        DROP TABLE IF EXISTS service_categories;
        DROP TABLE IF EXISTS product_categories;
        ALTER TABLE documents
          DROP COLUMN IF EXISTS document_type_confidence,
          DROP COLUMN IF EXISTS detected_document_type;
        """
    )
