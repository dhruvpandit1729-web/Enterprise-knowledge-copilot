from __future__ import annotations

from pathlib import Path
import re

from app.ingestion.base import BaseParser
from app.ingestion.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    UnreadableDocumentError,
)
from app.ingestion.normalizer import normalize_text
from app.schemas.document import ExtractedDocument, ExtractedSection


class MarkdownParser(BaseParser):
    """Parser for Markdown documents using UTF-8 extraction and heading detection."""

    @property
    def supported_extensions(self) -> set[str]:
        return {".md", ".markdown"}

    def parse(self, file_path: str | Path) -> ExtractedDocument:
        path = Path(file_path).resolve()
        if not path.exists():
            raise DocumentNotFoundError(f"File not found: '{path}'")
        if not path.is_file():
            raise UnreadableDocumentError(f"Path is not a regular file: '{path}'")
        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"Markdown file '{path.name}' is empty (0 bytes).")

        try:
            with open(path, "rb") as f:
                content = f.read()
        except OSError as exc:
            raise UnreadableDocumentError(f"Unable to read Markdown file '{path.name}': {exc}") from exc

        return self.parse_bytes(content, filename=path.name)

    def parse_bytes(self, content: bytes, filename: str) -> ExtractedDocument:
        if not content:
            raise EmptyDocumentError(f"Markdown file '{filename}' is empty (0 bytes).")

        try:
            raw_text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UnreadableDocumentError(
                f"Failed to decode Markdown file '{filename}' with UTF-8: {exc}"
            ) from exc

        if not raw_text.strip():
            raise EmptyDocumentError(
                f"Markdown file '{filename}' is empty or contains only whitespace."
            )

        sections: list[ExtractedSection] = []
        heading_stack: dict[int, str] = {}
        current_heading_path: str | None = None
        current_block_lines: list[str] = []

        def _flush_current_block() -> None:
            nonlocal current_block_lines
            if not current_block_lines:
                return
            combined = "\n".join(current_block_lines)
            normalized = normalize_text(combined)
            if normalized:
                sections.append(
                    ExtractedSection(
                        text=normalized,
                        page_number=None,  # Do not invent page numbers
                        heading=current_heading_path,
                        metadata={
                            "heading": current_heading_path,
                        }
                        if current_heading_path
                        else {},
                    )
                )
            current_block_lines = []

        lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        in_code_block = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check fenced code block toggle
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_block = not in_code_block
                current_block_lines.append(line)
                i += 1
                continue

            if in_code_block:
                current_block_lines.append(line)
                i += 1
                continue

            # Check ATX heading: # Heading
            atx_match = re.match(r"^(#{1,6})\s+(.+?)(?:\s+#+)?$", stripped)
            if atx_match:
                level = len(atx_match.group(1))
                heading_title = normalize_text(atx_match.group(2))

                _flush_current_block()

                heading_stack = {lvl: text for lvl, text in heading_stack.items() if lvl < level}
                heading_stack[level] = heading_title
                current_heading_path = " > ".join(
                    heading_stack[lvl] for lvl in sorted(heading_stack.keys())
                )
                current_block_lines.append(line)
                i += 1
                continue

            # Check Setext heading: Heading \n ===== or -----
            if (
                i + 1 < len(lines)
                and stripped
                and not stripped.startswith(("-", "*", "+", ">", "|", "#"))
            ):
                next_stripped = lines[i + 1].strip()
                if (
                    re.match(r"^(=+|-+)$", next_stripped)
                    and len(next_stripped) >= 2
                ):
                    level = 1 if next_stripped.startswith("=") else 2
                    heading_title = normalize_text(stripped)

                    _flush_current_block()

                    heading_stack = {lvl: text for lvl, text in heading_stack.items() if lvl < level}
                    heading_stack[level] = heading_title
                    current_heading_path = " > ".join(
                        heading_stack[lvl] for lvl in sorted(heading_stack.keys())
                    )
                    current_block_lines.append(line)
                    # Advance past the setext underline line
                    i += 2
                    continue

            # Regular line
            current_block_lines.append(line)
            i += 1

        # Flush any remaining block
        _flush_current_block()

        if not sections:
            raise EmptyDocumentError(
                f"Markdown file '{filename}' contains no extractable text."
            )

        return ExtractedDocument(
            source=filename,
            sections=sections,
            metadata={
                "doc_type": "markdown",
                "section_count": len(sections),
            },
        )
