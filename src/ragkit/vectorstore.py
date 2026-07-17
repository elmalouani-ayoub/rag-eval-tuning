"""Magasin vectoriel minimal.

Implementation par defaut : numpy pur (produit scalaire sur vecteurs L2-normalises
= similarite cosinus). Le corpus etant petit, c'est exact, rapide et sans dependance.

Une variante FAISS est fournie pour montrer la transition vers un index scalable
(memes resultats, indispensable quand le corpus grossit).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


class NumpyVectorStore:
    """Recherche exacte par similarite cosinus (brute force)."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._chunks = chunks
        self._matrix = np.asarray(vectors, dtype="float32")

    def search(self, query_vec: np.ndarray, top_k: int) -> list[Retrieved]:
        if self._matrix is None:
            raise RuntimeError("Index vide : appeler add() avant search().")
        query_vec = np.asarray(query_vec, dtype="float32").reshape(-1)
        scores = self._matrix @ query_vec  # cosinus car vecteurs L2-normalises
        k = min(top_k, len(self._chunks))
        top_idx = np.argsort(-scores)[:k]
        return [Retrieved(chunk=self._chunks[i], score=float(scores[i])) for i in top_idx]


class FaissVectorStore:
    """Meme interface, backend FAISS (index plat, produit scalaire).

    Necessite `pip install faiss-cpu`. Fournie pour illustrer la scalabilite ;
    non utilisee dans le sweep hors-ligne.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._index = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        import faiss

        vectors = np.asarray(vectors, dtype="float32")
        self._chunks = chunks
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[Retrieved]:
        if self._index is None:
            raise RuntimeError("Index vide : appeler add() avant search().")
        query_vec = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        k = min(top_k, len(self._chunks))
        scores, idx = self._index.search(query_vec, k)
        return [
            Retrieved(chunk=self._chunks[i], score=float(s))
            for i, s in zip(idx[0], scores[0])
        ]


def get_vectorstore(backend: str = "numpy"):
    backend = backend.lower()
    if backend == "numpy":
        return NumpyVectorStore()
    if backend == "faiss":
        return FaissVectorStore()
    raise ValueError(f"Backend vectorstore inconnu : {backend!r}")
