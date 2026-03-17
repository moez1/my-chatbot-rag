"""Document parsing: extract raw text from different file formats.

This is the FIRST step of the ingestion pipeline:
    File (PDF, .py, .md, .txt) → raw text string

Each format needs its own extraction logic:
- PDF: uses pypdf to read each page
- Code/Text/Markdown: reads the file as UTF-8
"""

import hashlib
from pathlib import Path

from pypdf import PdfReader


# Supported file extensions grouped by type
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".kt",
    ".sh", ".bash", ".sql", ".tf", ".hcl",
}
TEXT_EXTENSIONS = {
    ".txt", ".md", ".yaml", ".yml", ".json", ".toml",
    ".cfg", ".ini", ".dockerfile", ".html", ".css",
    ".xml", ".csv", ".rst",
}


def parse_file(file_path: Path) -> tuple[str, str]:
    """Extract text from a file and compute its content hash.

    Args:
        file_path: Path to the file to parse.

    Returns:
        Tuple of (extracted_text, sha256_hash).

    Raises:
        ValueError: If the file format is not supported.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        text = _parse_pdf(file_path)
    elif suffix in CODE_EXTENSIONS | TEXT_EXTENSIONS:
        text = _parse_text(file_path)
    else:
        msg = f"Unsupported file format: {suffix}"
        raise ValueError(msg)

    content_hash = hashlib.sha256(text.encode()).hexdigest()
    return text, content_hash


def _parse_pdf(file_path: Path) -> str:
    """Extract text from a PDF file, page by page."""
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages)


def _parse_text(file_path: Path) -> str:
    """Read a text/code file as UTF-8."""
    return file_path.read_text(encoding="utf-8")


def detect_source_type(file_path: Path) -> str:
    """Detect the source type from file extension.

    Returns:
        A category: pdf, code, markdown, or text.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".md":
        return "markdown"
    if suffix in CODE_EXTENSIONS:
        return "code"
    return "text"
