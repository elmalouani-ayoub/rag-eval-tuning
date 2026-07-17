"""Configuration : une seule combinaison d'hyperparametres = un `RunConfig`.

La grille de recherche (le "sweep") est decrite dans config.yaml et transformee
en une liste de RunConfig par `expand_grid`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    """Une combinaison d'hyperparametres a evaluer."""

    chunk_size: int          # taille de chunk (en nombre de mots)
    chunk_overlap: int       # recouvrement entre chunks consecutifs (en mots)
    top_k: int               # nombre de chunks recuperes au retrieval
    embedding: str           # methode d'embedding : "tfidf", "hashing", "sentence-transformers", "ollama"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def label(self) -> str:
        return (
            f"emb={self.embedding}|size={self.chunk_size}"
            f"|ov={self.chunk_overlap}|k={self.top_k}"
        )


def load_grid(config_path: str | Path) -> dict[str, Any]:
    """Charge le fichier YAML de configuration du sweep."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_grid(grid: dict[str, Any]) -> list[RunConfig]:
    """Produit le produit cartesien de toutes les valeurs d'hyperparametres.

    On ecarte les combinaisons incoherentes (overlap >= chunk_size).
    """
    g = grid["grid"]
    combos = itertools.product(
        g["chunk_size"],
        g["chunk_overlap"],
        g["top_k"],
        g["embedding"],
    )
    configs: list[RunConfig] = []
    for size, overlap, k, emb in combos:
        if overlap >= size:
            continue  # combinaison invalide, on saute
        configs.append(
            RunConfig(chunk_size=size, chunk_overlap=overlap, top_k=k, embedding=emb)
        )
    return configs
