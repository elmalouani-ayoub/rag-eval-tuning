"""Variante OPTIONNELLE du pipeline construite avec LangChain + Chroma + Ollama.

Le pipeline principal (`pipeline.py`) est volontairement framework-agnostic : cela
donne un controle total sur l'evaluation et evite la fragilite des versions de
frameworks. Ce module montre la meme logique cablee avec LangChain, pour les postes
qui demandent explicitement cette stack (ex. offre RAG du Ministere de l'Interieur).

Prerequis :
    pip install langchain langchain-community langchain-ollama langchain-chroma chromadb
    ollama pull mistral
    ollama pull nomic-embed-text

Non couvert par la CI (dependances lourdes + serveur Ollama requis).
"""
from __future__ import annotations

from pathlib import Path


def build_langchain_rag(
    corpus_dir: str | Path,
    chunk_size: int = 600,       # en caracteres pour le splitter LangChain
    chunk_overlap: int = 100,
    top_k: int = 3,
    llm_model: str = "mistral",
    embed_model: str = "nomic-embed-text",
):
    """Construit une chaine RAG LangChain et renvoie une fonction `ask(question)`."""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    # 1. Chargement + chunking
    loader = DirectoryLoader(str(corpus_dir), glob="*.md", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    # 2. Embeddings + index vectoriel (Chroma, en memoire)
    embeddings = OllamaEmbeddings(model=embed_model)
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 3. Prompt + LLM
    prompt = ChatPromptTemplate.from_template(
        "Reponds a la question en t'appuyant STRICTEMENT sur le contexte.\n"
        "Si l'info n'y est pas, dis 'Je ne sais pas'.\n\n"
        "Contexte:\n{context}\n\nQuestion: {question}\n\nReponse:"
    )
    llm = OllamaLLM(model=llm_model, temperature=0.0)

    def _format(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | _format, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    def ask(question: str) -> str:
        return chain.invoke(question)

    return ask


if __name__ == "__main__":
    import sys

    ask = build_langchain_rag(Path(__file__).resolve().parents[2] / "data" / "corpus")
    q = sys.argv[1] if len(sys.argv) > 1 else "Combien de jours de RTT par an ?"
    print(ask(q))
