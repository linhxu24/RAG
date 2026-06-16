from dataclasses import dataclass
from typing import Any

from app.ingestion.docling_parser import ParsedTextBlock


@dataclass
class TextChunk:
    content: str
    page_number: int | None
    section_title: str | None
    content_type: str = "text"
    metadata: dict[str, Any] | None = None


class DocumentChunker:
    def __init__(self, chunk_size: int = 1_200, chunk_overlap: int = 150):
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        except ImportError:
            self.splitter = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, blocks: list[ParsedTextBlock]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for block in blocks:
            pieces = (
                self.splitter.split_text(block.text)
                if self.splitter is not None
                else self._fallback_split(block.text)
            )
            chunks.extend(
                TextChunk(
                    content=piece.strip(),
                    page_number=block.page_number,
                    section_title=block.section_title,
                    metadata=block.metadata,
                )
                for piece in pieces
                if piece.strip()
            )
        return chunks

    def _fallback_split(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[index : index + self.chunk_size] for index in range(0, len(text), step)]
