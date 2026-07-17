"""Tests unitaires du pipeline et des metriques (executables en CI, sans modele)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragkit.chunking import chunk_document
from ragkit.config import RunConfig, expand_grid
from ragkit.loaders import Document, load_corpus, load_eval
from ragkit.metrics import evaluate_question, normalize
from ragkit.pipeline import RagPipeline
from ragkit.vectorstore import Retrieved
from ragkit.chunking import Chunk


# ----------------------------- chunking ----------------------------- #
def test_chunk_overlap_respected():
    doc = Document("d.md", " ".join(str(i) for i in range(100)))
    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)
    # avec step = 8, on couvre les 100 mots
    assert len(chunks) >= 12
    # le recouvrement : la fin d'un chunk se retrouve au debut du suivant
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-2:] == second_words[:2]


def test_chunk_overlap_invalid():
    doc = Document("d.md", "a b c d e")
    try:
        chunk_document(doc, chunk_size=5, chunk_overlap=5)
        assert False, "devrait lever ValueError"
    except ValueError:
        pass


# ----------------------------- normalize ----------------------------- #
def test_normalize_strips_accents_and_markdown():
    assert normalize("**Congés Payés**") == "conges payes"
    assert normalize("12   caractères") == "12 caracteres"


# ----------------------------- metrics ----------------------------- #
def _mk(doc_id):
    return Retrieved(chunk=Chunk(f"{doc_id}#0", doc_id, "texte 25 jours ouvres"), score=1.0)


def test_metrics_hit_and_mrr():
    from ragkit.loaders import EvalQuestion
    q = EvalQuestion("q", "?", "25 jours", "25 jours ouvres", "bon.md")
    retrieved = [_mk("mauvais.md"), _mk("bon.md")]
    r = evaluate_question(q, retrieved)
    assert r.hit == 1.0
    assert abs(r.reciprocal_rank - 0.5) < 1e-9  # bon doc en 2e position
    assert r.span_recall == 1.0


# ----------------------------- config ----------------------------- #
def test_expand_grid_filters_invalid():
    grid = {"grid": {"chunk_size": [10, 20], "chunk_overlap": [0, 10],
                     "top_k": [3], "embedding": ["tfidf"]}}
    configs = expand_grid(grid)
    # overlap 10 >= size 10 est ecarte
    assert all(c.chunk_overlap < c.chunk_size for c in configs)


# ----------------------------- pipeline end-to-end ----------------------------- #
def test_pipeline_end_to_end_offline():
    docs = load_corpus(ROOT / "data" / "corpus")
    cfg = RunConfig(chunk_size=80, chunk_overlap=20, top_k=3, embedding="tfidf")
    pipe = RagPipeline(docs, cfg)
    ans = pipe.answer("Combien de jours de teletravail par semaine ?")
    assert len(ans.retrieved) == 3
    # le bon document doit remonter
    assert any("teletravail" in r.chunk.doc_id for r in ans.retrieved)


def test_eval_set_wellformed():
    qs = load_eval(ROOT / "data" / "eval" / "questions.json")
    assert len(qs) == 20
    docs = {d.doc_id for d in load_corpus(ROOT / "data" / "corpus")}
    for q in qs:
        assert q.source_doc in docs, f"{q.id} pointe vers un doc inexistant"
