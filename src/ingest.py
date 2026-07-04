"""Builds the BM25 and FAISS indexes from the Apex Developer Guide PDF.

Run with: python -m src.ingest
"""

import pickle
import time

import pdfplumber
import tiktoken
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from voyageai.error import RateLimitError

from src import config

load_dotenv()

# Voyage's no-payment-method tier caps requests at 3 RPM / 10K TPM. Batch
# conservatively under that so ingestion works regardless of billing status,
# and only slow down for real if a batch still gets rate-limited.
MAX_BATCH_TOKENS = 9000
MIN_SECONDS_BETWEEN_BATCHES = 21
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _batch_by_token_budget(texts: list[str]) -> list[list[int]]:
    """Returns lists of indices into `texts`, each batch under the token budget."""
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for i, text in enumerate(texts):
        tokens = _token_count(text)
        if current and current_tokens + tokens > MAX_BATCH_TOKENS:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(i)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def _embed_with_rate_limit(
    embeddings: VoyageAIEmbeddings, texts: list[str]
) -> list[list[float]]:
    batches = _batch_by_token_budget(texts)
    vectors: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]

    for batch_num, indices in enumerate(batches, start=1):
        batch_texts = [texts[i] for i in indices]
        print(f"  embedding batch {batch_num}/{len(batches)} ({len(batch_texts)} chunks)...")

        for attempt in range(5):
            try:
                batch_vectors = embeddings.embed_documents(batch_texts)
                break
            except RateLimitError:
                wait = 65
                print(f"    rate-limited, waiting {wait}s before retrying...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Batch {batch_num} failed after repeated rate-limit retries")

        for i, vector in zip(indices, batch_vectors):
            vectors[i] = vector

        if batch_num < len(batches):
            time.sleep(MIN_SECONDS_BETWEEN_BATCHES)

    return vectors


def load_pages() -> list[Document]:
    """One Document per PDF page, tagged with page number and section."""
    documents = []
    with pdfplumber.open(config.PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages):
            page_number = i + 1
            text = page.extract_text()
            if not text or not text.strip():
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_number,
                        "section": config.section_for_page(page_number),
                    },
                )
            )
    return documents


def chunk_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)


def build_indexes(chunks: list[Document]) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    bm25 = BM25Retriever.from_documents(chunks)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    embeddings = VoyageAIEmbeddings(model=config.VOYAGE_MODEL)
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    vectors = _embed_with_rate_limit(embeddings, texts)

    vectorstore = FAISS.from_embeddings(
        list(zip(texts, vectors)), embeddings, metadatas=metadatas
    )
    vectorstore.save_local(str(config.FAISS_DIR))

    with open(config.CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def main() -> None:
    print(f"Parsing {config.PDF_PATH.name}...")
    pages = load_pages()
    print(f"Extracted {len(pages)} non-empty pages.")

    chunks = chunk_pages(pages)
    print(f"Split into {len(chunks)} chunks.")

    print("Building BM25 and FAISS indexes (this calls the Voyage AI embeddings API)...")
    build_indexes(chunks)
    print(f"Indexes written to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
