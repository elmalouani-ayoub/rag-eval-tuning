"""Metriques d'evaluation du RETRIEVAL (le coeur du projet).

Ces metriques ne dependent PAS du LLM : elles mesurent uniquement la qualite des
chunks recuperes par rapport a une verite-terrain. C'est ce qui permet d'evaluer
l'impact des hyperparametres (chunk_size, overlap, top_k, embedding) de facon
totalement reproductible, sans GPU ni acces reseau.

Verites-terrain utilisees :
  - `source_doc`  : le document qui contient la reponse    -> hit@k, MRR, precision@k
  - `answer_span` : un extrait verbatim attendu de la reponse -> span_recall@k

Pour la GENERATION, on fournit en option des metriques facon RAGAS
(fidelite, pertinence) qui, elles, necessitent un LLM (voir `ragas_evaluate`).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .loaders import EvalQuestion
from .vectorstore import Retrieved

_MARKDOWN_CHARS = re.compile(r"[*_`#~>]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """minuscules + suppression des accents + suppression du markdown + espaces compactes.

    Rend le matching de spans robuste (le corpus est accentue et contient du markdown,
    la verite-terrain non)."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _MARKDOWN_CHARS.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


@dataclass
class QuestionResult:
    id: str
    hit: float            # le bon document est-il dans le top_k ? (0/1)
    reciprocal_rank: float  # 1/rang du premier bon document (0 si absent)
    span_recall: float    # l'extrait attendu est-il dans un chunk recupere ? (0/1)
    precision: float      # part des chunks recuperes issus du bon document


def evaluate_question(q: EvalQuestion, retrieved: list[Retrieved]) -> QuestionResult:
    retrieved_docs = [r.chunk.doc_id for r in retrieved]

    # hit@k
    hit = 1.0 if q.source_doc in retrieved_docs else 0.0

    # MRR : rang (1-indexe) du premier chunk issu du bon document
    rr = 0.0
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id == q.source_doc:
            rr = 1.0 / rank
            break

    # precision@k : proportion de chunks recuperes issus du bon document
    precision = (
        sum(1 for d in retrieved_docs if d == q.source_doc) / len(retrieved_docs)
        if retrieved_docs
        else 0.0
    )

    # span_recall@k : l'extrait de reponse attendu apparait-il dans un chunk recupere ?
    span_norm = normalize(q.answer_span)
    joined = normalize(" ".join(r.chunk.text for r in retrieved))
    span_recall = 1.0 if span_norm in joined else 0.0

    return QuestionResult(
        id=q.id,
        hit=hit,
        reciprocal_rank=rr,
        span_recall=span_recall,
        precision=precision,
    )


@dataclass
class AggregateMetrics:
    hit_at_k: float
    mrr: float
    span_recall_at_k: float
    precision_at_k: float
    n_questions: int

    def as_dict(self) -> dict[str, float]:
        return {
            "hit_at_k": round(self.hit_at_k, 4),
            "mrr": round(self.mrr, 4),
            "span_recall_at_k": round(self.span_recall_at_k, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "n_questions": self.n_questions,
        }


def aggregate(results: list[QuestionResult]) -> AggregateMetrics:
    n = len(results)
    if n == 0:
        return AggregateMetrics(0, 0, 0, 0, 0)
    return AggregateMetrics(
        hit_at_k=sum(r.hit for r in results) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        span_recall_at_k=sum(r.span_recall for r in results) / n,
        precision_at_k=sum(r.precision for r in results) / n,
        n_questions=n,
    )


# --------------------------------------------------------------------------- #
# Metriques de GENERATION facon RAGAS (optionnel, necessite un LLM)
# --------------------------------------------------------------------------- #
def ragas_evaluate(pipeline, questions, judge_llm=None):
    """Evaluation de la GENERATION facon RAGAS (fidelite / pertinence).

    Necessite un LLM juge (Ollama). Non utilise dans le sweep hors-ligne ; fourni
    pour completer l'evaluation quand Ollama est disponible sur ta machine.

    Retourne, par question, un score de fidelite (la reponse est-elle etayee par le
    contexte ?) et de pertinence (repond-elle a la question ?), evalues par le LLM juge.
    """
    from .llm import get_llm

    judge = judge_llm or get_llm("ollama")
    rows = []
    for q in questions:
        ans = pipeline.answer(q.question)
        faith_prompt = (
            f"Contexte:\n{ans.context}\n\nReponse:\n{ans.answer}\n\n"
            "La reponse est-elle entierement etayee par le contexte ? "
            "Reponds uniquement par un score entre 0 et 1."
        )
        rel_prompt = (
            f"Question:\n{q.question}\n\nReponse:\n{ans.answer}\n\n"
            "La reponse repond-elle directement a la question ? "
            "Reponds uniquement par un score entre 0 et 1."
        )
        faith = _parse_score(judge.generate("", faith_prompt))
        rel = _parse_score(judge.generate("", rel_prompt))
        rows.append({"id": q.id, "faithfulness": faith, "answer_relevance": rel})
    return rows


def _parse_score(text: str) -> float:
    m = re.search(r"[01](?:[.,]\d+)?", text)
    if not m:
        return 0.0
    return float(m.group(0).replace(",", "."))
