import re
from dataclasses import dataclass, field

from sqlalchemy import func, literal, or_, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.constants import Intent
from app.db.models import Product, ProductAlias, Service, ServiceAlias
from app.retrieval.normalization import normalize_vietnamese, query_tokens


@dataclass
class EntityCandidate:
    entity_type: str
    entity_id: str
    name: str
    score: float
    match_type: str

    def as_dict(self) -> dict[str, object]:
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "name": self.name,
            "score": self.score,
            "match_type": self.match_type,
        }


@dataclass
class EntityResolution:
    status: str = "not_applicable"
    products: list[EntityCandidate] = field(default_factory=list)
    services: list[EntityCandidate] = field(default_factory=list)
    ambiguous_candidates: list[EntityCandidate] = field(default_factory=list)
    candidates: list[EntityCandidate] = field(default_factory=list)

    @property
    def selected(self) -> list[EntityCandidate]:
        return [*self.products, *self.services]

    @property
    def names(self) -> list[str]:
        return [item.name for item in self.selected]

    @property
    def best_score(self) -> float:
        return max((item.score for item in self.selected), default=0.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "products": [item.as_dict() for item in self.products],
            "services": [item.as_dict() for item in self.services],
            "ambiguous_candidates": [
                item.as_dict() for item in self.ambiguous_candidates
            ],
            "candidates": [item.as_dict() for item in self.candidates],
            "best_score": self.best_score,
        }


class DatabaseEntityResolver:
    def __init__(self, settings: Settings):
        self.settings = settings

    def resolve(
        self,
        session: Session,
        query: str,
        intent: Intent,
    ) -> EntityResolution:
        if intent not in {
            Intent.PRODUCT_DETAIL,
            Intent.PRODUCT_COMPARE,
            Intent.SERVICE_DETAIL,
        }:
            return EntityResolution()

        multiple = intent == Intent.PRODUCT_COMPARE
        segments = self._segments(query) if multiple else [query]
        entity_type = "service" if intent == Intent.SERVICE_DETAIL else "product"
        selected: list[EntityCandidate] = []
        ambiguous: list[EntityCandidate] = []
        all_candidates: list[EntityCandidate] = []
        for segment in segments:
            candidates = self._dedupe_by_name(
                self._candidates(session, segment, entity_type)
            )
            all_candidates.extend(candidates)
            if not candidates:
                continue
            selected.append(candidates[0])
            if (
                len(candidates) > 1
                and candidates[0].score - candidates[1].score
                < self.settings.entity_ambiguity_margin
            ):
                ambiguous.extend(candidates[:2])

        selected = self._dedupe(selected)
        expected_count = 2 if multiple else 1
        if ambiguous:
            status = "ambiguous"
        elif len(selected) >= expected_count:
            status = "resolved"
        elif selected:
            status = "partial"
        else:
            status = "not_found"
        return EntityResolution(
            status=status,
            products=selected if entity_type == "product" else [],
            services=selected if entity_type == "service" else [],
            ambiguous_candidates=self._dedupe(ambiguous),
            candidates=self._dedupe(all_candidates),
        )

    def _candidates(
        self,
        session: Session,
        query: str,
        entity_type: str,
    ) -> list[EntityCandidate]:
        model = Product if entity_type == "product" else Service
        id_column = Product.product_id if entity_type == "product" else Service.service_id
        normalized_query = normalize_vietnamese(query)
        if not normalized_query:
            return []
        normalized_name = func.lower(func.simplydent_unaccent(model.name))
        query_value = literal(normalized_query)
        contained = func.strpos(query_value, normalized_name) > 0
        distinctive_tokens = [
            token
            for token in query_tokens(query)
            if len(token) >= 6
        ]
        token_match = (
            or_(*(normalized_name.contains(token) for token in distinctive_tokens))
            if distinctive_tokens
            else literal(False)
        )
        similarity_score = func.greatest(
            func.similarity(normalized_name, query_value),
            func.word_similarity(normalized_name, query_value),
            func.word_similarity(query_value, normalized_name),
        )
        rows = session.execute(
            select(
                id_column,
                model.name,
                similarity_score.label("score"),
                contained.label("contained"),
                token_match.label("token_match"),
            )
            .where(
                model.status == "active",
                or_(
                    contained,
                    token_match,
                    similarity_score >= self.settings.entity_match_threshold,
                ),
            )
            .order_by(
                contained.desc(),
                token_match.desc(),
                similarity_score.desc(),
                model.name,
            )
            .limit(5)
        ).all()

        query_token_set = set(query_tokens(query))
        candidates: list[EntityCandidate] = []
        for entity_id, name, score, is_contained, is_token_match in rows:
            overlap = len(query_token_set & set(query_tokens(name)))
            if (
                not is_contained
                and not is_token_match
                and overlap < self._minimum_overlap(name)
            ):
                continue
            effective_score = (
                1.0
                if is_contained
                else max(0.88 if is_token_match else 0.0, float(score))
            )
            candidates.append(
                EntityCandidate(
                    entity_type=entity_type,
                    entity_id=str(entity_id),
                    name=str(name),
                    score=effective_score,
                    match_type=(
                        "contained"
                        if is_contained
                        else "token"
                        if is_token_match
                        else "trigram"
                    ),
                )
            )
        candidates.extend(self._alias_candidates(session, query_value, entity_type))
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def _alias_candidates(
        self,
        session: Session,
        query_value,
        entity_type: str,
    ) -> list[EntityCandidate]:
        model = Product if entity_type == "product" else Service
        alias_model = ProductAlias if entity_type == "product" else ServiceAlias
        id_column = Product.product_id if entity_type == "product" else Service.service_id
        fk_column = (
            ProductAlias.product_id
            if entity_type == "product"
            else ServiceAlias.service_id
        )
        alias_name = func.lower(func.simplydent_unaccent(alias_model.alias))
        alias_contained = func.strpos(query_value, alias_name) > 0
        alias_score = func.greatest(
            func.similarity(alias_name, query_value),
            func.word_similarity(alias_name, query_value),
            func.word_similarity(query_value, alias_name),
        )
        rows = session.execute(
            select(
                id_column,
                model.name,
                alias_score.label("score"),
                alias_contained.label("contained"),
            )
            .join(alias_model, fk_column == id_column)
            .where(
                model.status == "active",
                or_(
                    alias_contained,
                    alias_score >= self.settings.entity_match_threshold,
                ),
            )
            .order_by(alias_contained.desc(), alias_score.desc())
            .limit(5)
        ).all()
        return [
            EntityCandidate(
                entity_type=entity_type,
                entity_id=str(entity_id),
                name=str(name),
                score=1.0 if is_contained else float(score),
                match_type="alias",
            )
            for entity_id, name, score, is_contained in rows
        ]

    @staticmethod
    def _segments(query: str) -> list[str]:
        values = re.split(
            r"\s+(?:và|voi|với|vs\.?|so với|hay|khác)\s+",
            query,
            flags=re.IGNORECASE,
        )
        return [value.strip() for value in values if value.strip()]

    @staticmethod
    def _minimum_overlap(name: str) -> int:
        return 1 if len(query_tokens(name)) <= 2 else 2

    @staticmethod
    def _dedupe(values: list[EntityCandidate]) -> list[EntityCandidate]:
        return list({item.entity_id: item for item in values}.values())

    @staticmethod
    def _dedupe_by_name(values: list[EntityCandidate]) -> list[EntityCandidate]:
        seen: set[str] = set()
        deduplicated: list[EntityCandidate] = []
        for item in values:
            key = normalize_vietnamese(item.name)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(item)
        return deduplicated
