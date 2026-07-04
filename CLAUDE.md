# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Terminal QA agent that answers questions about Salesforce Apex grounded in `salesforce_apex_developer_guide.pdf` (821 pages), with guardrails to refuse off-topic questions.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY`. Voyage AI requires a payment method on file to get standard rate limits (otherwise capped at 3 RPM / 10K TPM); the free 200M-token allotment for `voyage-3` still applies either way, so this project's ~680K-token corpus costs effectively $0 to embed.

## Building the index

```
python -m src.ingest
```

Parses the PDF, chunks it, and writes `index/bm25.pkl`, `index/faiss/`, and `index/chunks.pkl` — gitignored build artifacts, not source. Re-run after changing chunking or the embedding model in `config.py`. Embedding calls are batched (`_embed_with_rate_limit` in `ingest.py`) to stay under Voyage's rate limits regardless of account tier; expect several minutes.

## Running the agent

```
python main.py
```

Plain terminal REPL: type a question, get a concise answer with page citations, type `exit`/`quit` to stop.

## Architecture

Query flow, per `src/cli.py`:

1. **`router.py`** — embeds the question and scores it against the 6 top-level section descriptions (hardcoded in `config.py` from the guide's own table of contents). This score is used only as a soft re-ranking boost, not a hard filter — the guide has frequent cross-references between sections (e.g. "see X on page Y"), so hard-restricting retrieval to one routed section risks starving legitimate cross-cutting questions.
2. **`retrieval.py`** — hybrid retrieval: BM25 keyword search + FAISS vector search, fused via a hand-rolled reciprocal rank fusion (RRF) rather than LangChain's `EnsembleRetriever`, so per-candidate scores stay available for re-ranking and guardrail use. Fused scores get boosted by the router's section score, then sorted for the top-K sent to the LLM.
3. **`guardrails.py`** — before any Claude call, checks the *raw* FAISS cosine relevance score (not the fused/boosted score — see below) against `config.RETRIEVAL_SCORE_THRESHOLD`; below it, the question is rejected as out of scope without an LLM call.
4. **`qa_chain.py`** — assembles the top chunks (with page/section metadata) into a prompt instructing Claude to answer only from that context, cite pages inline, and say so if the context doesn't actually cover the question — a second, softer guardrail layer as backstop.

### Why the router isn't a guardrail

The original design used the router's section-confidence score as a first guardrail layer (reject if the question doesn't resemble any section). Empirically this didn't discriminate at all — e.g. "what's the weather like today?" scored *higher* than a genuine cross-section Apex question ("how do governor limits interact with test methods?"). It was dropped as a guardrail and kept only for the soft re-ranking boost, where it still helps.

### Why the guardrail uses a separate raw score, not the retrieval-ranking score

`ScoredChunk.score` in `retrieval.py` is a fused RRF score normalized by the max score *within that query's own result set* — so the top hit is always close to 1.0 regardless of whether anything actually relevant was found (BM25/vector search always return *some* nearest match, however weak). Guardrail checks need an absolute, cross-query-comparable signal instead, so `retrieve()` separately returns the raw FAISS relevance score of the single closest chunk in the whole corpus, which is what `guardrails.py` actually checks. `RETRIEVAL_SCORE_THRESHOLD` (`config.py`) was set empirically to `0.3` after testing: real Apex questions scored 0.39–0.48, off-topic ones scored -0.11–0.23.

## Known limitations (MVP scope)

- Chunking doesn't guarantee code blocks stay intact — it relies on generous chunk size/overlap (`CHUNK_SIZE`/`CHUNK_OVERLAP` in `config.py`) rather than true layout-aware PDF parsing.
- Top-level section boundaries in `config.py` are hardcoded from the guide's TOC rather than parsed dynamically — fine as long as this stays pinned to the Summer '26 edition of the guide.
- `langchain-community` (used for `BM25Retriever` and `FAISS`) is in deprecation/sunset upstream; a future pass could migrate to standalone integration packages.

## Separate, unresolved issue (not part of this project)

This repo's git root is currently the user's home directory (`$HOME`), not `apex_llm/`. Recommend scoping a proper git repo to `apex_llm/` before committing any of this work.
