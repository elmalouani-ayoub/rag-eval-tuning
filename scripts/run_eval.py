#!/usr/bin/env python
"""Lance le balayage d'hyperparametres et ecrit results/results.csv.

Usage :
    python scripts/run_eval.py                 # utilise config.yaml
    python scripts/run_eval.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Rendre le package importable sans installation
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragkit.config import expand_grid, load_grid  # noqa: E402
from ragkit.evaluate import run_sweep  # noqa: E402
from ragkit.loaders import load_corpus, load_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep d'hyperparametres RAG")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    grid = load_grid(args.config)
    corpus_path = ROOT / grid["paths"]["corpus"]
    eval_path = ROOT / grid["paths"]["eval"]
    results_path = ROOT / grid["paths"]["results"]

    docs = load_corpus(corpus_path)
    questions = load_eval(eval_path)
    configs = expand_grid(grid)

    print(f"Corpus       : {len(docs)} documents")
    print(f"Evaluation   : {len(questions)} questions")
    print(f"Configurations a evaluer : {len(configs)}\n")

    df = run_sweep(docs, questions, configs)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"\nResultats ecrits dans : {results_path}")

    best = df.sort_values(["mrr", "hit_at_k"], ascending=False).iloc[0]
    print("\n=== Meilleure configuration (par MRR) ===")
    print(best.to_string())


if __name__ == "__main__":
    main()
