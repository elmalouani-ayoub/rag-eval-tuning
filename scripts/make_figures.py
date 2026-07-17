#!/usr/bin/env python
"""Genere les figures d'analyse a partir de results/results.csv.

Produit dans results/figures/ :
  1. embedding_comparison.png   - tfidf vs hashing (toutes metriques)
  2. topk_tradeoff.png          - compromis precision / rappel selon top_k
  3. chunk_size_effect.png      - effet de la taille de chunk (sweet spot)
  4. heatmap_size_overlap.png   - MRR selon chunk_size x chunk_overlap
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
BLUE, ORANGE, GREEN, RED = "#2563eb", "#ea580c", "#16a34a", "#dc2626"


def load() -> pd.DataFrame:
    path = ROOT / "results" / "results.csv"
    if not path.exists():
        sys.exit("results/results.csv introuvable. Lance d'abord: python scripts/run_eval.py")
    return pd.read_csv(path)


def fig_embedding(df: pd.DataFrame) -> None:
    metrics = ["hit_at_k", "mrr", "span_recall_at_k", "precision_at_k"]
    labels = ["hit@k", "MRR", "span_recall@k", "precision@k"]
    means = df.groupby("embedding")[metrics].mean()
    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, means.loc["tfidf"], w, label="tfidf", color=BLUE)
    ax.bar(x + w / 2, means.loc["hashing"], w, label="hashing", color=ORANGE)
    for i, m in enumerate(metrics):
        ax.text(i - w / 2, means.loc["tfidf", m] + 0.01, f"{means.loc['tfidf', m]:.2f}",
                ha="center", fontsize=9)
        ax.text(i + w / 2, means.loc["hashing", m] + 0.01, f"{means.loc['hashing', m]:.2f}",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score moyen (toutes configs)")
    ax.set_title("Methode d'embedding : TF-IDF vs Hashing")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "embedding_comparison.png"); plt.close(fig)


def fig_topk(df: pd.DataFrame) -> None:
    d = df[df.embedding == "tfidf"].groupby("top_k")[["mrr", "hit_at_k", "precision_at_k"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d.index, d["hit_at_k"], "o-", color=GREEN, label="hit@k (rappel doc)")
    ax.plot(d.index, d["mrr"], "s-", color=BLUE, label="MRR")
    ax.plot(d.index, d["precision_at_k"], "^-", color=RED, label="precision@k")
    ax.set_xlabel("top_k (nombre de chunks recuperes)")
    ax.set_ylabel("Score moyen (tfidf)")
    ax.set_xticks(d.index); ax.set_ylim(0, 1.05)
    ax.set_title("Compromis precision / rappel selon top_k")
    ax.legend()
    ax.annotate("plus de k = meilleur rappel\nmais precision qui chute",
                xy=(10, d["precision_at_k"].iloc[-1]), xytext=(6, 0.55),
                arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9, color="grey")
    fig.tight_layout(); fig.savefig(FIG / "topk_tradeoff.png"); plt.close(fig)


def fig_chunksize(df: pd.DataFrame) -> None:
    d = (df[(df.embedding == "tfidf") & (df.top_k >= 3)]
         .groupby("chunk_size")[["mrr", "span_recall_at_k"]].mean())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d.index, d["mrr"], "s-", color=BLUE, label="MRR (specificite du chunk)")
    ax.plot(d.index, d["span_recall_at_k"], "o-", color=GREEN,
            label="span_recall@k (couverture reponse)")
    ax.set_xlabel("Taille de chunk (mots)")
    ax.set_ylabel("Score moyen (tfidf, k>=3)")
    ax.set_ylim(0.9, 1.02)
    ax.set_title("Effet de la taille de chunk : un compromis")
    ax.legend()
    best = d["mrr"].idxmax()
    ax.axvline(best, color="grey", ls="--", alpha=0.6)
    ax.text(best, 0.995, f" meilleur MRR\n a {best} mots", fontsize=9, color="grey")
    fig.tight_layout(); fig.savefig(FIG / "chunk_size_effect.png"); plt.close(fig)


def fig_heatmap(df: pd.DataFrame) -> None:
    d = df[(df.embedding == "tfidf") & (df.top_k == 3)]
    piv = d.pivot_table(index="chunk_size", columns="chunk_overlap", values="mrr")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto", vmin=0.9, vmax=1.0)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("chunk_overlap (mots)"); ax.set_ylabel("chunk_size (mots)")
    ax.set_title("MRR selon chunk_size x chunk_overlap (tfidf, k=3)")
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.97 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="MRR")
    fig.tight_layout(); fig.savefig(FIG / "heatmap_size_overlap.png"); plt.close(fig)


def main() -> None:
    df = load()
    fig_embedding(df)
    fig_topk(df)
    fig_chunksize(df)
    fig_heatmap(df)
    print(f"4 figures ecrites dans {FIG}")


if __name__ == "__main__":
    main()
