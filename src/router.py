"""Soft coarse-section router.

Embeds the guide's 6 top-level section descriptions once, then scores an
incoming question against each via cosine similarity. Scores are used as a
soft boost during retrieval re-ranking, not as a hard filter, and the best
score doubles as guardrail layer 1 (reject if nothing looks relevant).
"""

import numpy as np
from langchain_voyageai import VoyageAIEmbeddings

from src import config


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


class SectionRouter:
    def __init__(self) -> None:
        self._embeddings = VoyageAIEmbeddings(model=config.VOYAGE_MODEL)
        self._section_names = [s[0] for s in config.TOP_LEVEL_SECTIONS]
        descriptions = [s[3] for s in config.TOP_LEVEL_SECTIONS]
        self._section_vectors = self._embeddings.embed_documents(descriptions)

    def score_sections(self, question: str) -> dict[str, float]:
        """Cosine similarity of the question against each section description."""
        question_vector = self._embeddings.embed_query(question)
        return {
            name: _cosine_similarity(question_vector, vector)
            for name, vector in zip(self._section_names, self._section_vectors)
        }

    def best_score(self, section_scores: dict[str, float]) -> float:
        return max(section_scores.values(), default=0.0)
