"""Pipeline RAG : chunking -> embeddings -> index vectoriel -> retrieval -> generation.

Construit a partir d'un `RunConfig`. Volontairement simple et lisible : le but du
projet est l'EVALUATION, pas un pipeline sophistique. La conception reste toutefois
compatible LangChain/LlamaIndex (memes briques, memes interfaces).
"""
from __future__ import annotations

from dataclasses import dataclass

from .chunking import chunk_corpus, Chunk
from .config import RunConfig
from .embeddings import get_embedder
from .llm import get_llm, LLM
from .loaders import Document
from .vectorstore import get_vectorstore, Retrieved


@dataclass
class RagAnswer:
    question: str
    answer: str
    retrieved: list[Retrieved]

    @property
    def context(self) -> str:
        return "\n\n".join(r.chunk.text for r in self.retrieved)


class RagPipeline:
    def __init__(
        self,
        docs: list[Document],
        config: RunConfig,
        llm: LLM | None = None,
        vectorstore_backend: str = "numpy",
    ) -> None:
        self.config = config
        self.llm = llm  # peut rester None si on ne fait que du retrieval/eval

        # 1. Chunking
        self.chunks: list[Chunk] = chunk_corpus(
            docs, config.chunk_size, config.chunk_overlap
        )

        # 2. Embeddings (fit sur le corpus pour tfidf)
        self.embedder = get_embedder(config.embedding)
        self.embedder.fit([c.text for c in self.chunks])
        vectors = self.embedder.encode([c.text for c in self.chunks])

        # 3. Index vectoriel
        self.store = get_vectorstore(vectorstore_backend)
        self.store.add(self.chunks, vectors)

    def retrieve(self, question: str) -> list[Retrieved]:
        q_vec = self.embedder.encode([question])[0]
        return self.store.search(q_vec, top_k=self.config.top_k)

    def answer(self, question: str) -> RagAnswer:
        retrieved = self.retrieve(question)
        if self.llm is None:
            self.llm = get_llm("extractive")
        context = "\n\n".join(r.chunk.text for r in retrieved)
        text = self.llm.generate(question, context)
        return RagAnswer(question=question, answer=text, retrieved=retrieved)
