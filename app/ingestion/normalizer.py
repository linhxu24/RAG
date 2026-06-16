import re

from app.ingestion.docling_parser import ParsedDocument, ParsedTextBlock

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(
        r"\[asset:([^\]]+)\]",
        lambda match: "[asset:" + match.group(1).replace("\\_", "_") + "]",
        text,
    )
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize_document(document: ParsedDocument) -> ParsedDocument:
    normalized_blocks: list[ParsedTextBlock] = []
    seen: set[tuple[str, int | None, str | None]] = set()
    for block_index, block in enumerate(document.text_blocks):
        for section_index, section in enumerate(_split_markdown_sections(block)):
            text = normalize_whitespace(section.text)
            key = (text, section.page_number, section.section_title)
            if not text or key in seen:
                continue
            seen.add(key)
            normalized_blocks.append(
                ParsedTextBlock(
                    text=text,
                    page_number=section.page_number,
                    section_title=section.section_title,
                    metadata={
                        **block.metadata,
                        **section.metadata,
                        "source_block_index": block_index,
                        "section_index": section_index,
                    },
                )
            )
    document.text_blocks = normalized_blocks
    document.tables = [table for table in document.tables if table.rows]
    document.assets = [asset for asset in document.assets if asset.data]
    document.metadata["normalized_counts"] = {
        "text_blocks": len(document.text_blocks),
        "table_blocks": len(document.tables),
        "image_blocks": len(document.assets),
    }
    return document


def _split_markdown_sections(block: ParsedTextBlock) -> list[ParsedTextBlock]:
    sections: list[ParsedTextBlock] = []
    current_title = block.section_title
    current_lines: list[str] = []

    def flush() -> None:
        if not current_lines:
            return
        sections.append(
            ParsedTextBlock(
                text="\n".join(current_lines),
                page_number=block.page_number,
                section_title=current_title,
                metadata={"normalized_from_markdown_section": True},
            )
        )
        current_lines.clear()

    for line in block.text.splitlines():
        heading = MARKDOWN_HEADING.match(line.strip())
        if heading:
            flush()
            current_title = heading.group(2).strip()
            current_lines.append(line)
        else:
            current_lines.append(line)
    flush()
    return sections or [block]
