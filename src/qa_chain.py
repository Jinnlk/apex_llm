"""Prompt assembly and Claude call for answering from retrieved context."""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src import config
from src.retrieval import ScoredChunk

SYSTEM_PROMPT = (
    "You are a QA assistant for the Salesforce Apex Developer Guide. "
    "Answer ONLY using the excerpts provided below, which are taken from the "
    "guide. Be concise. Cite the page number(s) you drew from inline, like "
    "(p. 42). If the excerpts don't actually answer the question, say the "
    "guide doesn't seem to cover it rather than guessing or using outside "
    "knowledge."
)


def _format_context(chunks: list[ScoredChunk]) -> str:
    parts = []
    for chunk in chunks:
        page = chunk.document.metadata.get("page")
        section = chunk.document.metadata.get("section")
        parts.append(f"[p. {page} — {section}]\n{chunk.document.page_content}")
    return "\n\n---\n\n".join(parts)


class QAChain:
    def __init__(self) -> None:
        self._llm = ChatAnthropic(model=config.ANTHROPIC_MODEL)

    def answer(self, question: str, chunks: list[ScoredChunk]) -> tuple[str, list]:
        context = _format_context(chunks)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Excerpts:\n\n{context}\n\nQuestion: {question}"
            ),
        ]
        response = self._llm.invoke(messages)

        sources = sorted(
            {
                (c.document.metadata.get("page"), c.document.metadata.get("section"))
                for c in chunks
            }
        )
        return response.content, sources
