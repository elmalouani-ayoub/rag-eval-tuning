"""Embeddings pluggables.

Quatre backends, meme interface (`fit` + `encode`) :

  - "tfidf"                 : sklearn TF-IDF        -> hors-ligne, aucun telechargement (defaut CI/demo)
  - "hashing"              : sklearn HashingVectorizer -> hors-ligne, sert de 2e methode a comparer
  - "sentence-transformers": modele HuggingFace     -> qualite semantique, telecharge le modele
  - "ollama"               : embeddings via Ollama   -> chemin "production" local (ex. nomic-embed-text)

Les deux premiers permettent de faire tourner TOUT le harnais d'evaluation sans GPU
ni acces reseau. Les deux derniers donnent de meilleurs resultats semantiques et
sont a activer sur ta machine.

Toutes les sorties sont normalisees en L2 : la similarite cosinus se reduit alors a
un simple produit scalaire (voir vectorstore.py).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> "Embedder": ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Backends hors-ligne (aucun telechargement)
# --------------------------------------------------------------------------- #
class TfidfEmbedder:
    """TF-IDF classique. Le vocabulaire est appris sur le corpus (fit)."""

    def __init__(self, ngram_max: int = 1) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, ngram_max),
            min_df=1,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self._vec.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Appeler fit() sur le corpus avant encode().")
        mat = self._vec.transform(texts).toarray().astype("float32")
        return _l2_normalize(mat)


class HashingEmbedder:
    """Hashing trick : sans vocabulaire, stateless. Utile comme methode de comparaison."""

    def __init__(self, n_features: int = 4096) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(
            lowercase=True,
            strip_accents="unicode",
            n_features=n_features,
            alternate_sign=False,
            norm=None,
        )

    def fit(self, texts: list[str]) -> "HashingEmbedder":
        return self  # stateless

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = self._vec.transform(texts).toarray().astype("float32")
        return _l2_normalize(mat)


# --------------------------------------------------------------------------- #
# Backends "production" (a activer sur ta machine)
# --------------------------------------------------------------------------- #
class SentenceTransformerEmbedder:
    """Embeddings semantiques via HuggingFace sentence-transformers.

    Modele par defaut multilingue leger. Necessite `pip install sentence-transformers`
    et un premier telechargement du modele.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype="float32")


class OllamaEmbedder:
    """Embeddings via un serveur Ollama local (http://localhost:11434).

    Modele conseille : `ollama pull nomic-embed-text`.
    C'est le chemin 100 % local et gratuit decrit dans le plan portfolio.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")

    def fit(self, texts: list[str]) -> "OllamaEmbedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        import requests

        vecs = []
        for text in texts:
            resp = requests.post(
                f"{self._host}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=120,
            )
            resp.raise_for_status()
            vecs.append(resp.json()["embedding"])
        return _l2_normalize(np.asarray(vecs, dtype="float32"))


# --------------------------------------------------------------------------- #
# Fabrique
# --------------------------------------------------------------------------- #
def get_embedder(name: str, **kwargs) -> Embedder:
    name = name.lower()
    if name == "tfidf":
        return TfidfEmbedder(**kwargs)
    if name == "hashing":
        return HashingEmbedder(**kwargs)
    if name in ("sentence-transformers", "st", "hf"):
        return SentenceTransformerEmbedder(**kwargs)
    if name == "ollama":
        return OllamaEmbedder(**kwargs)
    raise ValueError(
        f"Embedding inconnu : {name!r}. "
        "Choisir parmi : tfidf, hashing, sentence-transformers, ollama."
    )
