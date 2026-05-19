import hashlib
from dataclasses import dataclass
from pathlib import Path

import docx2txt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _load_pdf(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((idx, text))
    return pages


def _load_docx(path: Path) -> list[tuple[int, str]]:
    text = (docx2txt.process(str(path)) or "").strip()
    if not text:
        return []
    return [(1, text)]


def load_document(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".docx", ".doc"}:
        return _load_docx(path)
    if suffix == ".txt":
        return [(1, path.read_text(encoding="utf-8"))]
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_document(
    pages: list[tuple[int, str]],
    source_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    out: list[Chunk] = []
    for page_num, page_text in pages:
        for chunk_text in splitter.split_text(page_text):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            out.append(
                Chunk(
                    text=chunk_text,
                    source=source_name,
                    page=page_num,
                    chunk_id=_hash(f"{source_name}:{page_num}:{chunk_text}"),
                )
            )
    return out
