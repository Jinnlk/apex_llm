"""Hybrid retrieval: BM25 + FAISS fused via reciprocal rank fusion (RRF),
then re-ranked with a soft boost from the section router.

This hand-rolls the same weighted RRF that LangChain's EnsembleRetriever
uses internally, so per-candidate scores are available for the guardrail
threshold check and the section boost, rather than staying opaque inside
that wrapper.
"""

import pickle
import warnings
from typing import NamedTuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings

from src import config
from src.router import SectionRouter

RRF_K = 60


class ScoredChunk(NamedTuple):
    document: Document
    score: float


def _doc_key(doc: Document) -> tuple:
    return (doc.metadata.get("page"), doc.page_content[:80])


def _rrf_contribution(ranked_docs: list[Document], weight: float) -> dict:
    return {
        _doc_key(doc): weight / (RRF_K + rank + 1)
        for rank, doc in enumerate(ranked_docs)
    }


class HybridRetriever:
    def __init__(self, router: SectionRouter) -> None:
        with open(config.BM25_PATH, "rb") as f:
            self._bm25 = pickle.load(f)
        self._bm25.k = config.RETRIEVAL_TOP_K * 4

        embeddings = VoyageAIEmbeddings(model=config.VOYAGE_MODEL)
        self._vectorstore = FAISS.load_local(
            str(config.FAISS_DIR), embeddings, allow_dangerous_deserialization=True
        )
        self._router = router

    def retrieve(self, question: str) -> tuple[list[ScoredChunk], dict, float]:
        """Returns (top chunks for the prompt, section scores, absolute relevance).

        `absolute_relevance` is the raw cosine similarity of the question against
        its single closest chunk in the whole corpus. Unlike the fused/boosted
        `ScoredChunk.score` values below (which are normalized per-query and so
        always put the top hit near 1.0 regardless of how relevant it actually
        is), this score is comparable across queries and is what the guardrail
        threshold check should use.
        """
        section_scores = self._router.score_sections(question)

        bm25_docs = self._bm25.invoke(question)
        vector_docs = self._vectorstore.similarity_search(
            question, k=config.RETRIEVAL_TOP_K * 4
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            relevance_results = self._vectorstore.similarity_search_with_relevance_scores(
                question, k=1
            )
        absolute_relevance = relevance_results[0][1] if relevance_results else 0.0

        fused: dict = {}
        for key, contribution in _rrf_contribution(bm25_docs, config.BM25_WEIGHT).items():
            fused[key] = fused.get(key, 0.0) + contribution
        for key, contribution in _rrf_contribution(vector_docs, config.VECTOR_WEIGHT).items():
            fused[key] = fused.get(key, 0.0) + contribution

        by_key = {_doc_key(d): d for d in [*bm25_docs, *vector_docs]}
        max_fused = max(fused.values(), default=0.0) or 1.0

        boosted: list[ScoredChunk] = []
        for key, fused_score in fused.items():
            doc = by_key[key]
            normalized = fused_score / max_fused
            boost = config.SECTION_BOOST_WEIGHT * section_scores.get(
                doc.metadata.get("section"), 0.0
            )
            boosted.append(ScoredChunk(document=doc, score=normalized + boost))

        boosted.sort(key=lambda c: c.score, reverse=True)
        return boosted[: config.RETRIEVAL_TOP_K], section_scores, absolute_relevance
