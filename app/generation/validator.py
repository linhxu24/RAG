import json
import re
import uuid
from copy import deepcopy
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Asset, Chunk, Document, Product, Service, TableRow
from app.generation.schemas import GeneratedResponse


class ResponseValidationError(ValueError):
    pass


class ResponseValidator:
    def validate(
        self,
        payload: str | dict[str, Any],
        *,
        context: dict[str, Any],
        session: Session | None = None,
    ) -> GeneratedResponse:
        try:
            data = json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError as exc:
            raise ResponseValidationError(f"Invalid JSON: {exc}") from exc
        data = self._normalize_server_managed_assets(data, context)
        try:
            response = GeneratedResponse.model_validate(data)
        except Exception as exc:
            raise ResponseValidationError(f"Schema validation failed: {exc}") from exc
        allowed_ids = {
            str(item.get("source_id")) for item in context.get("items", []) if item.get("source_id")
        }
        allowed_doc_ids = {
            str(item.get("source", {}).get("doc_id"))
            for item in context.get("items", [])
            if item.get("source", {}).get("doc_id")
        }
        allowed_asset_ids = {
            str(item.get("raw_json", {}).get("asset_id"))
            for item in context.get("items", [])
            if item.get("raw_json", {}).get("asset_id")
        }
        for item in response.result.items:
            for value in (item.id, item.chunk_id, item.row_id):
                if value and value not in allowed_ids and not self._exists(session, value):
                    raise ResponseValidationError(f"Referenced ID does not exist: {value}")
            if (
                item.doc_id
                and item.doc_id not in allowed_doc_ids
                and not self._exists(session, item.doc_id)
            ):
                raise ResponseValidationError(f"Referenced doc_id does not exist: {item.doc_id}")
            for asset_id in item.asset_ids:
                if asset_id not in allowed_asset_ids and not self._exists(session, asset_id):
                    raise ResponseValidationError(f"Referenced asset_id does not exist: {asset_id}")
        for entity in response.entities:
            if (
                entity.matched_id
                and entity.matched_id not in allowed_ids
                and not self._exists(session, entity.matched_id)
            ):
                raise ResponseValidationError(
                    f"Referenced entity ID does not exist: {entity.matched_id}"
                )
        for source in response.result.sources:
            if source.source_id not in allowed_ids and not self._exists(session, source.source_id):
                raise ResponseValidationError(
                    f"Referenced source ID does not exist: {source.source_id}"
                )
            if (
                source.doc_id
                and source.doc_id not in allowed_doc_ids
                and not self._exists(session, source.doc_id)
            ):
                raise ResponseValidationError(
                    f"Referenced source doc_id does not exist: {source.doc_id}"
                )
        for asset in response.result.assets:
            asset_id = asset.get("asset_id")
            if asset_id and not self._exists(session, str(asset_id)):
                raise ResponseValidationError(
                    f"Referenced response asset_id does not exist: {asset_id}"
                )
        self._validate_prices(response.result.text, context)
        return response

    @staticmethod
    def _normalize_server_managed_assets(
        data: Any,
        context: dict[str, Any],
    ) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = deepcopy(data)
        result = normalized.get("result")
        if not isinstance(result, dict):
            return normalized

        # Asset resolution is deterministic and happens after generation. The model must
        # not invent resolved or missing asset objects.
        result["assets"] = []
        result["missing_assets"] = []

        asset_aliases: dict[str, str] = {}
        for context_item in context.get("items", []):
            raw_json = context_item.get("raw_json") or {}
            asset_id = raw_json.get("asset_id")
            if not asset_id:
                continue
            asset_id = str(asset_id)
            asset_aliases[asset_id] = asset_id
            for token in re.findall(r"\[asset:([^\]]+)\]", str(context_item.get("text") or "")):
                asset_aliases[token] = asset_id
                asset_aliases[f"[asset:{token}]"] = asset_id

        items = result.get("items")
        if not isinstance(items, list):
            return normalized
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("asset_ids"), list):
                continue
            resolved_ids: list[str] = []
            for value in item["asset_ids"]:
                if isinstance(value, dict):
                    value = value.get("asset_id") or value.get("token") or value.get("id")
                if value is None:
                    continue
                candidate = str(value)
                mapped = asset_aliases.get(candidate)
                if mapped and mapped not in resolved_ids:
                    resolved_ids.append(mapped)
            item["asset_ids"] = resolved_ids
        return normalized

    @staticmethod
    def _validate_prices(text: str, context: dict[str, Any]) -> None:
        generated_prices = ResponseValidator._price_numbers(text)
        if not generated_prices:
            return
        context_text = json.dumps(context, ensure_ascii=False, default=str)
        context_numbers = ResponseValidator._all_numbers(context_text)
        unsupported = [price for price in generated_prices if price not in context_numbers]
        if unsupported:
            raise ResponseValidationError(f"Unsupported price value(s): {unsupported}")

    @staticmethod
    def _price_numbers(text: str) -> set[int]:
        values: set[int] = set()
        patterns = (
            r"(?:giá|chi phí)\s*(?:là|:)?\s*([\d.,]+)",
            r"([\d.,]+)\s*(?:đ|₫|vnd)\b",
        )
        for pattern in patterns:
            for match in re.findall(pattern, text.lower()):
                normalized = ResponseValidator._normalize_numeric(match)
                if normalized is not None:
                    values.add(normalized)
        return values

    @staticmethod
    def _all_numbers(text: str) -> set[int]:
        values: set[int] = set()
        for match in re.findall(r"\d[\d.,]*", text):
            normalized = ResponseValidator._normalize_numeric(match)
            if normalized is not None:
                values.add(normalized)
        return values

    @staticmethod
    def _normalize_numeric(value: str) -> int | None:
        value = value.strip().rstrip(".,")
        if not value:
            return None
        if re.search(r"[.,]\d{2}$", value) and not re.search(r"[.,]\d{3}[.,]\d{2}$", value):
            value = value[:-3]
        digits = re.sub(r"\D", "", value)
        return int(digits) if digits else None

    @staticmethod
    def _exists(session: Session | None, value: str) -> bool:
        if session is None:
            return False
        try:
            identifier = uuid.UUID(value)
        except (ValueError, TypeError):
            return False
        for _model, column in (
            (Chunk, Chunk.chunk_id),
            (Asset, Asset.asset_id),
            (Product, Product.product_id),
            (Service, Service.service_id),
            (Document, Document.doc_id),
            (TableRow, TableRow.row_id),
        ):
            if session.scalar(select(column).where(column == identifier)) is not None:
                return True
        return False
