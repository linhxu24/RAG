import json
import re
import uuid
from copy import deepcopy
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Asset, Chunk, Document, Product, Service, TableRow
from app.generation.schemas import GeneratedResponse
from app.retrieval.normalization import normalize_vietnamese


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
        data = self._normalize_context_references(data, context)
        try:
            response = GeneratedResponse.model_validate(data)
        except Exception as exc:
            raise ResponseValidationError(f"Schema validation failed: {exc}") from exc
        allowed_ids = {
            str(item.get("source_id")) for item in context.get("items", []) if item.get("source_id")
        }
        allowed_ids.update(
            str(value)
            for item in context.get("items", [])
            for value in (
                item.get("source", {}).get("row_id"),
                item.get("source", {}).get("table_id"),
            )
            if value
        )
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
        self._validate_semantics(response.result.text, context)
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
    def _normalize_context_references(
        data: Any,
        context: dict[str, Any],
    ) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = deepcopy(data)
        context_items = context.get("items", [])
        if not isinstance(context_items, list):
            return normalized
        alias_to_item: dict[str, dict[str, Any]] = {}
        allowed_ids: set[str] = set()
        allowed_doc_ids: set[str] = set()
        allowed_row_ids: set[str] = set()
        for context_item in context_items:
            source_id = context_item.get("source_id")
            if not source_id:
                continue
            source_id = str(source_id)
            allowed_ids.add(source_id)
            source = context_item.get("source") or {}
            if source.get("doc_id"):
                allowed_doc_ids.add(str(source["doc_id"]))
            if source.get("row_id"):
                allowed_row_ids.add(str(source["row_id"]))
            aliases = {
                source_id,
                str(context_item.get("canonical_key") or ""),
            }
            raw_json = context_item.get("raw_json") or {}
            for key in ("name", "question", "key"):
                value = raw_json.get(key)
                if value:
                    aliases.add(str(value))
                    aliases.add(ResponseValidator._reference_alias(str(value)))
                    aliases.add(
                        f"{context_item.get('source_type')}:{ResponseValidator._reference_alias(str(value))}"
                    )
            for alias in aliases:
                if alias:
                    alias_to_item[alias] = context_item

        result = normalized.get("result")
        if not isinstance(result, dict):
            return normalized

        items = result.get("items")
        if isinstance(items, list):
            normalized_items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                context_item = ResponseValidator._context_item_for_reference(
                    item.get("id"),
                    item.get("name"),
                    item.get("type"),
                    alias_to_item,
                    allowed_ids,
                )
                if context_item is None:
                    continue
                source = context_item.get("source") or {}
                raw_json = context_item.get("raw_json") or {}
                item["id"] = str(context_item["source_id"])
                item["type"] = item.get("type") or context_item.get("source_type")
                item["name"] = item.get("name") or raw_json.get("name") or raw_json.get("question")
                if item.get("chunk_id") not in allowed_ids:
                    item["chunk_id"] = (
                        item["id"] if context_item.get("source_type") == "chunk" else None
                    )
                if item.get("row_id") not in allowed_ids | allowed_row_ids:
                    item["row_id"] = source.get("row_id")
                if item.get("doc_id") not in allowed_doc_ids:
                    item["doc_id"] = source.get("doc_id")
                normalized_items.append(item)
            result["items"] = normalized_items

        sources = result.get("sources")
        if isinstance(sources, list):
            normalized_sources = []
            for source_ref in sources:
                if not isinstance(source_ref, dict):
                    continue
                context_item = ResponseValidator._context_item_for_reference(
                    source_ref.get("source_id"),
                    None,
                    source_ref.get("source_type"),
                    alias_to_item,
                    allowed_ids,
                )
                if context_item is None:
                    continue
                source = context_item.get("source") or {}
                source_ref["source_id"] = str(context_item["source_id"])
                source_ref["source_type"] = (
                    source_ref.get("source_type") or context_item.get("source_type")
                )
                if source_ref.get("doc_id") not in allowed_doc_ids:
                    source_ref["doc_id"] = source.get("doc_id")
                normalized_sources.append(source_ref)
            result["sources"] = normalized_sources

        entities = normalized.get("entities")
        if isinstance(entities, list):
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                context_item = ResponseValidator._context_item_for_reference(
                    entity.get("matched_id"),
                    entity.get("name"),
                    entity.get("type"),
                    alias_to_item,
                    allowed_ids,
                )
                entity["matched_id"] = (
                    str(context_item["source_id"]) if context_item is not None else None
                )
        return normalized

    @staticmethod
    def _context_item_for_reference(
        identifier: Any,
        name: Any,
        source_type: Any,
        alias_to_item: dict[str, dict[str, Any]],
        allowed_ids: set[str],
    ) -> dict[str, Any] | None:
        candidates = [identifier, name]
        if source_type and name:
            candidates.append(f"{source_type}:{ResponseValidator._reference_alias(str(name))}")
        for candidate in candidates:
            if candidate is None:
                continue
            candidate = str(candidate)
            if candidate in allowed_ids and candidate in alias_to_item:
                return alias_to_item[candidate]
            if candidate in alias_to_item:
                return alias_to_item[candidate]
            normalized = ResponseValidator._reference_alias(candidate)
            if normalized in alias_to_item:
                return alias_to_item[normalized]
        return None

    @staticmethod
    def _reference_alias(value: str) -> str:
        return normalize_vietnamese(value)

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
    def _validate_semantics(text: str, context: dict[str, Any]) -> None:
        normalized_text = normalize_vietnamese(text)
        tasks = context.get("tasks")
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                intent = str(task.get("intent") or "")
                if intent not in {
                    "PRODUCT_DETAIL",
                    "PRODUCT_COMPARE",
                    "SERVICE_DETAIL",
                }:
                    continue
                names = task.get("entity_names")
                if not isinstance(names, list):
                    continue
                missing = [
                    str(name)
                    for name in names
                    if not ResponseValidator._entity_mentioned(
                        normalized_text,
                        str(name),
                    )
                ]
                if missing:
                    raise ResponseValidationError(
                        "Answer does not identify the canonical task entity: "
                        f"{missing}"
                    )

        duration_values = {
            int(item.get("raw_json", {}).get("duration_minutes"))
            for item in context.get("items", [])
            if _int_value(
                item.get("raw_json", {}).get("duration_minutes")
            )
            is not None
        }
        for item in context.get("items", []):
            duration_values.update(
                ResponseValidator._duration_minutes(
                    str(item.get("text") or "")
                )
            )
        generated_durations = ResponseValidator._duration_minutes(text)
        unsupported_durations = generated_durations - duration_values
        if unsupported_durations:
            raise ResponseValidationError(
                "Unsupported duration value(s): "
                f"{sorted(unsupported_durations)}"
            )

        product_quantities = [
            _int_value(item.get("raw_json", {}).get("quantity"))
            for item in context.get("items", [])
            if item.get("source_type") == "product"
        ]
        product_quantities = [
            value for value in product_quantities if value is not None
        ]
        normalized_context = normalize_vietnamese(
            json.dumps(context, ensure_ascii=False, default=str)
        )
        claims_out_of_stock = (
            "het hang" in normalized_text
            or "khong con hang" in normalized_text
        )
        claims_in_stock = (
            "con hang" in normalized_text
            and not claims_out_of_stock
        )
        if (
            claims_in_stock
            and "con hang" not in normalized_context
            and not any(quantity > 0 for quantity in product_quantities)
        ):
            raise ResponseValidationError(
                "Unsupported availability claim: còn hàng"
            )
        if (
            claims_out_of_stock
            and not any(
                phrase in normalized_context
                for phrase in ("het hang", "khong con hang")
            )
            and not any(quantity == 0 for quantity in product_quantities)
        ):
            raise ResponseValidationError(
                "Unsupported availability claim: hết hàng"
            )

        generated_times = ResponseValidator._clock_values(text)
        if generated_times:
            context_text = json.dumps(
                context,
                ensure_ascii=False,
                default=str,
            )
            missing_times = (
                generated_times
                - ResponseValidator._clock_values(context_text)
            )
            if missing_times:
                raise ResponseValidationError(
                    "Unsupported opening-hour value(s): "
                    f"{sorted(missing_times)}"
                )

    @staticmethod
    def _entity_mentioned(
        normalized_answer: str,
        entity_name: str,
    ) -> bool:
        normalized_name = normalize_vietnamese(entity_name)
        if not normalized_name:
            return True
        if normalized_name in normalized_answer:
            return True
        tokens = [
            token
            for token in normalized_name.split()
            if token not in {
                "product",
                "service",
                "dich",
                "vu",
                "rang",
                "tai",
                "phong",
                "kham",
            }
        ]
        if any(
            f"{left} {right}" in normalized_answer
            for left, right in zip(tokens, tokens[1:], strict=False)
        ):
            return True
        return any(
            len(token) >= 7 and token in normalized_answer
            for token in tokens
        )

    @staticmethod
    def _duration_minutes(text: str) -> set[int]:
        values: set[int] = set()
        normalized = normalize_vietnamese(text)
        for number, unit in re.findall(
            r"\b(\d+(?:[\.,]\d+)?)\s*(phut|gio)\b",
            normalized,
        ):
            try:
                value = float(number.replace(",", "."))
            except ValueError:
                continue
            values.add(
                int(value * 60)
                if unit == "gio"
                else int(value)
            )
        return values

    @staticmethod
    def _clock_values(text: str) -> set[int]:
        values: set[int] = set()
        for hour, minute in re.findall(
            r"\b(\d{1,2})(?::|h)(\d{2})\b",
            text.lower(),
        ):
            parsed_hour = int(hour)
            parsed_minute = int(minute)
            if 0 <= parsed_hour <= 23 and 0 <= parsed_minute <= 59:
                values.add(parsed_hour * 60 + parsed_minute)
        return values

    @staticmethod
    def _price_numbers(text: str) -> set[int]:
        values: set[int] = set()
        patterns = (
            r"(?:giá|chi phí)\s*(?:là|:)?\s*([\d.,]+)\s*(triệu|trieu|nghìn|nghin|k)?",
            r"([\d.,]+)\s*(triệu|trieu|nghìn|nghin|k|đ|₫|vnd)\b",
        )
        for pattern in patterns:
            for match in re.findall(pattern, text.lower()):
                if isinstance(match, tuple):
                    value, unit = match
                else:
                    value, unit = match, ""
                normalized = ResponseValidator._normalize_numeric(value, unit)
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
    def _normalize_numeric(value: str, unit: str = "") -> int | None:
        value = value.strip().rstrip(".,")
        if not value:
            return None
        decimal_value = None
        if unit in {"triệu", "trieu", "nghìn", "nghin", "k"}:
            try:
                decimal_value = float(value.replace(".", "").replace(",", "."))
            except ValueError:
                decimal_value = None
        if decimal_value is not None:
            if unit in {"triệu", "trieu"}:
                return int(decimal_value * 1_000_000)
            if unit in {"nghìn", "nghin", "k"}:
                return int(decimal_value * 1_000)
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


def _int_value(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
