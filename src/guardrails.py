"""Layered guardrail checks for rejecting out-of-scope questions.

Layer 1 (retrieval confidence) runs before any Claude call, so an obviously
off-topic question never reaches the LLM. Layer 2 (the system prompt's
refusal instruction, in qa_chain.py) is a backstop for cases where retrieval
finds something weakly relevant but the chunk doesn't actually answer the
question.

An earlier version also gated on the section router's confidence score, but
that score didn't discriminate on-topic from off-topic questions in testing
(see config.py) and was dropped as a guardrail -- the router is still used
for soft re-ranking in retrieval.py.
"""

from src import config

REFUSAL_MESSAGE = config.OUT_OF_SCOPE_MESSAGE


def passes_retrieval_confidence(absolute_relevance: float) -> bool:
    """Guardrail layer 1: did retrieval actually find something relevant?"""
    return absolute_relevance >= config.RETRIEVAL_SCORE_THRESHOLD
