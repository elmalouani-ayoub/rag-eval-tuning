"""Chargement du corpus documentaire et du jeu d'evaluation.

Le corpus est un dossier de fichiers .md (base de connaissances metier).
Chaque document est identifie par son nom de fichier (doc_id).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str   # nom du fichier, ex. "01-conges-rtt.md"
    text: str


@dataclass
class EvalQuestion:
    id: str
    question: str
    answer: str
    answer_span: str   # extrait verbatim attendu (verite-terrain span-level)
    source_doc: str    # doc_id contenant la reponse (verite-terrain retrieval)


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    corpus_dir = Path(corpus_dir)
    docs: list[Document] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(doc_id=path.name, text=text))
    if not docs:
        raise FileNotFoundError(f"Aucun document .md trouve dans {corpus_dir}")
    return docs


def load_eval(eval_path: str | Path) -> list[EvalQuestion]:
    data = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    return [
        EvalQuestion(
            id=q["id"],
            question=q["question"],
            answer=q["answer"],
            answer_span=q["answer_span"],
            source_doc=q["source_doc"],
        )
        for q in data["questions"]
    ]
