from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    source_type: str
    source_id: str
    text: str
    score: float
    raw_json: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    canonical_key: str | None = None

    @property
    def key(self) -> str:
        return self.canonical_key or f"{self.source_type}:{self.source_id}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "text": self.text,
            "score": self.score,
            "raw_json": self.raw_json,
            "source": self.source,
            "ranks": self.ranks,
            "canonical_key": self.canonical_key,
        }
