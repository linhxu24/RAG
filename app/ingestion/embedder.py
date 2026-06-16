import hashlib
import math
import re
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


class EmbeddingConfigurationError(RuntimeError):
    pass


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None
        self._model_load_attempted = False
        self._model_load_error: Exception | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, kind="document")

    def embed_query(self, text_value: str) -> list[float]:
        return self._embed([text_value], kind="query")[0]

    @property
    def backend_name(self) -> str:
        model = self._load_model()
        return (
            self.settings.embedding_model
            if model is not None
            else "deterministic_hash_fallback"
        )

    @property
    def using_fallback(self) -> bool:
        return self.backend_name == "deterministic_hash_fallback"

    @property
    def model_dimension(self) -> int:
        model = self._load_model()
        if model is None:
            return self.settings.embedding_dim
        dimension_getter = getattr(model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = model.get_sentence_embedding_dimension
        dimension = dimension_getter()
        if dimension is None:
            probe = model.encode(["dimension probe"], show_progress_bar=False)
            dimension = len(probe[0])
        return int(dimension)

    def validate_configuration(self, session: Session | None = None) -> dict[str, Any]:
        backend = self.backend_name
        if backend == "deterministic_hash_fallback" and not self.settings.allow_embedding_fallback:
            raise EmbeddingConfigurationError(
                "Embedding model is unavailable and ALLOW_EMBEDDING_FALLBACK=false"
            )
        model_dimension = self.model_dimension
        if model_dimension != self.settings.embedding_dim:
            raise EmbeddingConfigurationError(
                f"Embedding model dimension {model_dimension} does not match "
                f"EMBEDDING_DIM={self.settings.embedding_dim}"
            )

        database_dimensions: dict[str, int] = {}
        if session is not None:
            rows = session.execute(
                text(
                    """
                    SELECT c.relname AS table_name,
                           a.attname AS column_name,
                           format_type(a.atttypid, a.atttypmod) AS formatted_type
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = current_schema()
                      AND c.relname IN ('chunks', 'table_rows', 'faqs')
                      AND a.attname = 'embedding'
                      AND a.attnum > 0
                    ORDER BY c.relname
                    """
                )
            ).all()
            for table_name, column_name, formatted_type in rows:
                match = re.fullmatch(r"vector\((\d+)\)", formatted_type)
                if not match:
                    raise EmbeddingConfigurationError(
                        f"{table_name}.{column_name} has unexpected type {formatted_type}"
                    )
                dimension = int(match.group(1))
                database_dimensions[f"{table_name}.{column_name}"] = dimension
                if dimension != self.settings.embedding_dim:
                    raise EmbeddingConfigurationError(
                        f"{table_name}.{column_name} dimension {dimension} does not match "
                        f"EMBEDDING_DIM={self.settings.embedding_dim}"
                    )

        return {
            "backend": backend,
            "model": self.settings.embedding_model,
            "model_dimension": model_dimension,
            "configured_dimension": self.settings.embedding_dim,
            "database_dimensions": database_dimensions,
            "fallback": self.using_fallback,
        }

    def _embed(self, texts: list[str], kind: Literal["query", "document"]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        if model is not None:
            if self.model_dimension != self.settings.embedding_dim:
                raise EmbeddingConfigurationError(
                    f"Embedding model dimension {self.model_dimension} does not match "
                    f"EMBEDDING_DIM={self.settings.embedding_dim}"
                )
            prepared = self._prepare_for_model(texts, kind)
            vectors = model.encode(
                prepared,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            result = [vector.tolist() for vector in vectors]
            if any(len(vector) != self.settings.embedding_dim for vector in result):
                raise EmbeddingConfigurationError("Embedding model returned an invalid dimension")
            return result
        if not self.settings.allow_embedding_fallback:
            detail = f": {self._model_load_error}" if self._model_load_error else ""
            raise EmbeddingConfigurationError(
                f"Embedding model is unavailable and fallback is disabled{detail}"
            )
        return [self._hash_embedding(value) for value in texts]

    def _load_model(self):
        if self._model_load_attempted:
            if (
                self._model is None
                and self.settings.strict_embedding
                and not self.settings.allow_embedding_fallback
            ):
                raise EmbeddingConfigurationError(
                    f"Unable to load embedding model {self.settings.embedding_model}: "
                    f"{self._model_load_error}"
                )
            return self._model
        self._model_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.settings.embedding_device,
            )
        except Exception as exc:
            self._model_load_error = exc
            if self.settings.strict_embedding and not self.settings.allow_embedding_fallback:
                raise EmbeddingConfigurationError(
                    f"Unable to load embedding model {self.settings.embedding_model}: {exc}"
                ) from exc
            logger.warning(
                "Embedding model unavailable; deterministic fallback is enabled: %s", exc
            )
        return self._model

    def _prepare_for_model(self, texts: list[str], kind: str) -> list[str]:
        if "e5" in self.settings.embedding_model.lower():
            prefix = "query: " if kind == "query" else "passage: "
            return [prefix + value for value in texts]
        return texts

    def _hash_embedding(self, value: str) -> list[float]:
        vector = [0.0] * self.settings.embedding_dim
        tokens = re.findall(r"\w+", value.lower(), flags=re.UNICODE)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.settings.embedding_dim
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(item * item for item in vector)) or 1.0
        return [item / norm for item in vector]
