#!/usr/bin/env python
"""Interroge le RAG en ligne de commande (demo qualitative).

Exemples :
    # Hors-ligne (aucun modele) :
    python scripts/ask.py "Combien de jours de teletravail par semaine ?"

    # Avec Ollama (Mistral) et embeddings Ollama :
    python scripts/ask.py "..." --embedding ollama --llm ollama --model mistral
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragkit.config import RunConfig  # noqa: E402
from ragkit.llm import get_llm  # noqa: E402
from ragkit.loaders import load_corpus  # noqa: E402
from ragkit.pipeline import RagPipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--embedding", default="tfidf",
                   help="tfidf | hashing | sentence-transformers | ollama")
    p.add_argument("--llm", default="extractive", help="extractive | ollama")
    p.add_argument("--model", default="mistral", help="modele Ollama pour le LLM")
    p.add_argument("--chunk-size", type=int, default=120)
    p.add_argument("--chunk-overlap", type=int, default=20)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    docs = load_corpus(ROOT / "data" / "corpus")
    config = RunConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        embedding=args.embedding,
    )
    llm = get_llm(args.llm, model=args.model) if args.llm == "ollama" else get_llm(args.llm)
    pipeline = RagPipeline(docs, config, llm=llm)

    ans = pipeline.answer(args.question)
    print("\n=== Reponse ===")
    print(ans.answer)
    print("\n=== Sources (top_k) ===")
    for r in ans.retrieved:
        print(f"  - {r.chunk.chunk_id}  (score={r.score:.3f})")


if __name__ == "__main__":
    main()
