"""Decoupage des documents en chunks (fenetre glissante par mots).

On travaille en nombre de MOTS plutot qu'en caracteres : c'est plus lisible et
independant de la langue. `chunk_size` et `chunk_overlap` sont exprimes en mots.
"""
from __future__ import annotations

from dataclasses import dataclass

from .loaders import Document


@dataclass
class Chunk:
    chunk_id: str   # ex. "01-conges-rtt.md#3"
    doc_id: str     # document source
    text: str


def _split_words(text: str) -> list[str]:
    return text.split()


def chunk_document(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Decoupe un document en chunks avec recouvrement."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit etre strictement inferieur a chunk_size")

    words = _split_words(doc.text)
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    idx = 0
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        text = " ".join(window)
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}#{idx}", doc_id=doc.doc_id, text=text))
        idx += 1
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def chunk_corpus(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return chunks
