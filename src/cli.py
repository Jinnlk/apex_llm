"""Terminal REPL for the Apex Developer Guide QA agent."""

from dotenv import load_dotenv

from src import guardrails
from src.qa_chain import QAChain
from src.retrieval import HybridRetriever
from src.router import SectionRouter

load_dotenv()


def run() -> None:
    print("Apex Developer Guide QA Agent")
    print("Ask a question about Apex. Type 'exit' or 'quit' to stop.\n")

    router = SectionRouter()
    retriever = HybridRetriever(router)
    qa_chain = QAChain()

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        chunks, _section_scores, absolute_relevance = retriever.retrieve(question)

        if not chunks or not guardrails.passes_retrieval_confidence(absolute_relevance):
            print(guardrails.REFUSAL_MESSAGE + "\n")
            continue

        answer, sources = qa_chain.answer(question, chunks)
        print(answer)
        for page, section in sources:
            print(f'Source: p. {page}, "{section}"')
        print()


if __name__ == "__main__":
    run()
