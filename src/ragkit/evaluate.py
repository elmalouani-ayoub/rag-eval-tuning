"""Moteur de balayage (sweep) : evalue chaque combinaison d'hyperparametres.

Pour chaque `RunConfig` :
  1. construit le pipeline (chunking + embeddings + index) -- retrieval uniquement,
     aucun LLM requis ;
  2. execute les 20 questions du jeu d'evaluation ;
  3. agrege les metriques (hit@k, MRR, span_recall@k, precision@k).

Le resultat est un tableau (une ligne par configuration) exploite ensuite par le
notebook d'analyse et par les scripts de generation de figures.
"""
from __future__ import annotations

import pandas as pd

from .config import RunConfig
from .loaders import Document, EvalQuestion
from .metrics import aggregate, evaluate_question
from .pipeline import RagPipeline


def evaluate_config(
    docs: list[Document], questions: list[EvalQuestion], config: RunConfig
) -> dict:
    pipeline = RagPipeline(docs, config, llm=None)  # retrieval only
    results = [evaluate_question(q, pipeline.retrieve(q.question)) for q in questions]
    agg = aggregate(results)
    row = {
        "embedding": config.embedding,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "n_chunks": len(pipeline.chunks),
        **agg.as_dict(),
    }
    return row


def run_sweep(
    docs: list[Document],
    questions: list[EvalQuestion],
    configs: list[RunConfig],
    verbose: bool = True,
) -> pd.DataFrame:
    rows = []
    for i, cfg in enumerate(configs, start=1):
        row = evaluate_config(docs, questions, cfg)
        rows.append(row)
        if verbose:
            print(
                f"[{i:>3}/{len(configs)}] {cfg.label():45s} "
                f"hit@k={row['hit_at_k']:.3f}  mrr={row['mrr']:.3f}  "
                f"span={row['span_recall_at_k']:.3f}"
            )
    return pd.DataFrame(rows)
