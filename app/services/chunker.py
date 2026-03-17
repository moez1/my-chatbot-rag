"""Text chunking: split raw text into smaller pieces.

This is the SECOND step of the ingestion pipeline:
    raw text string → list of chunks

Why chunk?
- A PDF can be 50 pages = ~25,000 tokens
- LLM context is limited (and expensive to fill)
- Vector search is more precise on small pieces
- We only send the 3-5 most relevant chunks to the LLM

Strategy: fixed-size with overlap.
- chunk_size: max characters per chunk (default 500)
- chunk_overlap: characters shared between consecutive chunks (default 50)

The overlap ensures we don't lose context at boundaries.

Example (chunk_size=100, overlap=20):
    Chunk 0: characters   0-100
    Chunk 1: characters  80-180   (80-100 shared with chunk 0)
    Chunk 2: characters 160-260   (160-180 shared with chunk 1)
"""

from dataclasses import dataclass


@dataclass
class ChunkData:
    """A piece of text with its position in the original document."""

    index: int
    text: str
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[ChunkData]:
    """Split text into overlapping chunks.

    Args:
        text: The full text to split.
        chunk_size: Max characters per chunk.
        chunk_overlap: Characters shared between consecutive chunks.

    Returns:
        List of ChunkData with text and position info.
    """
    if not text.strip():
        return []

    chunks: list[ChunkData] = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size

        # Don't cut in the middle of a word — find the last space
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append(ChunkData(
                index=index,
                text=chunk_content,
                start_char=start,
                end_char=end,
            ))
            index += 1

        # Move forward, keeping overlap
        start = end - chunk_overlap
        if start >= end:
            start = end

    return chunks
