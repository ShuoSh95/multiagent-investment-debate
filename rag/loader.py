"""
Unified document loader: PDF / HTML / TXT / EPUB / Markdown -> list[Document].

Each Document carries metadata: {source, filename, master, source_type}.
Downstream chunker adds year, credibility_tier, etc.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document


# ============================================================
#  Individual format loaders
# ============================================================

def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_epub(path: Path) -> str:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    texts: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        texts.append(soup.get_text(separator="\n"))
    return "\n\n".join(texts)


def _load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


FORMAT_HANDLERS = {
    ".pdf": _load_pdf,
    ".epub": _load_epub,
    ".html": _load_html,
    ".htm": _load_html,
    ".txt": _load_text,
    ".md": _load_text,
    ".markdown": _load_text,
}

SUPPORTED_EXTENSIONS = set(FORMAT_HANDLERS.keys())


# ============================================================
#  Public API
# ============================================================

def load_file(
    path: Path,
    master: str,
    source_type: str = "book",
    extra_metadata: Optional[dict] = None,
) -> Document:
    """Load a single file and return one large Document (pre-chunking)."""
    ext = path.suffix.lower()
    handler = FORMAT_HANDLERS.get(ext)
    if handler is None:
        raise ValueError(f"Unsupported file format: {ext} ({path})")

    text = handler(path)
    text = _clean_text(text)

    metadata = {
        "source": str(path),
        "filename": path.name,
        "master": master,
        "source_type": source_type,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return Document(page_content=text, metadata=metadata)


def load_directory(
    directory: Path,
    master: str,
    source_type: str = "book",
    extra_metadata: Optional[dict] = None,
) -> List[Document]:
    """Load all supported files from a directory (non-recursive by default)."""
    if not directory.is_dir():
        return []

    docs: List[Document] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                doc = load_file(path, master, source_type, extra_metadata)
                if doc.page_content.strip():
                    docs.append(doc)
            except Exception as e:
                print(f"  [WARN] Skipping {path.name}: {e}")
    return docs


# ============================================================
#  Text cleaning
# ============================================================

def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
