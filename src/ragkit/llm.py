"""Generateurs de reponse pluggables.

  - "ollama"     : LLM open-weight local (Mistral 7B, Llama 3 8B...) via Ollama.
                   C'est le chemin decrit dans le plan portfolio (gratuit, 100 % local).
  - "extractive" : repli hors-ligne, sans modele. Renvoie le passage le plus pertinent.
                   Permet de faire tourner le pipeline de bout en bout en CI/demo.

Le harnais d'evaluation du RETRIEVAL ne depend d'aucun LLM (voir metrics.py) :
c'est ce qui rend le tuning d'hyperparametres reproductible sans GPU.
"""
from __future__ import annotations

from typing import Protocol

PROMPT_TEMPLATE = """Tu es un assistant qui repond STRICTEMENT a partir du contexte fourni.
Si la reponse n'est pas dans le contexte, reponds "Je ne sais pas".

Contexte :
{context}

Question : {question}

Reponse concise :"""


class LLM(Protocol):
    def generate(self, question: str, context: str) -> str: ...


class ExtractiveLLM:
    """Repli sans modele : renvoie la premiere phrase du contexte contenant
    le plus de mots de la question. Suffisant pour une demo de bout en bout."""

    def generate(self, question: str, context: str) -> str:
        import re

        q_words = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 3}
        sentences = re.split(r"(?<=[.!?])\s+", context)
        best, best_score = "", -1
        for sent in sentences:
            s_words = {w.lower() for w in re.findall(r"\w+", sent)}
            score = len(q_words & s_words)
            if score > best_score:
                best, best_score = sent, score
        return best.strip() or "Je ne sais pas"


class OllamaLLM:
    """Generation via un serveur Ollama local (http://localhost:11434).

    Prerequis :  ollama pull mistral   (ou llama3)
    """

    def __init__(
        self,
        model: str = "mistral",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._temperature = temperature

    def generate(self, question: str, context: str) -> str:
        import requests

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        resp = requests.post(
            f"{self._host}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self._temperature},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


def get_llm(name: str, **kwargs) -> LLM:
    name = name.lower()
    if name == "extractive":
        return ExtractiveLLM()
    if name == "ollama":
        return OllamaLLM(**kwargs)
    raise ValueError(f"LLM inconnu : {name!r}. Choisir parmi : extractive, ollama.")
