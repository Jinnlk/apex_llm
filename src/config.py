"""Static configuration for the Apex Developer Guide QA agent.

Section boundaries and descriptions are hardcoded from the PDF's own
table of contents (page 3) and overview page (page 5) rather than
derived via heading-detection, since the TOC is stable and small.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "salesforce_apex_developer_guide.pdf"
INDEX_DIR = PROJECT_ROOT / "index"
FAISS_DIR = INDEX_DIR / "faiss"
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
ANTHROPIC_MODEL = "claude-sonnet-5"
VOYAGE_MODEL = "voyage-3"

# (name, first_page, last_page, description) — last_page is inclusive.
# Page numbers are 1-indexed as printed in the document.
TOP_LEVEL_SECTIONS = [
    (
        "Getting Started with Apex",
        1,
        21,
        "Learn about the Apex development lifecycle. Follow a step-by-step "
        "tutorial to create an Apex class and trigger, and deploy them to a "
        "production organization.",
    ),
    (
        "Writing Apex",
        22,
        262,
        "Apex is like Java for Salesforce. It enables you to add and interact "
        "with data in the Lightning Platform persistence layer. It uses "
        "classes, data types, variables, and if-else statements. You can make "
        "it execute based on a condition, or have a block of code execute "
        "repeatedly.",
    ),
    (
        "Running Apex",
        263,
        676,
        "You can access many features of the Salesforce user interface "
        "programmatically in Apex, and you can integrate with external SOAP "
        "and REST Web services. You can run Apex code using a variety of "
        "mechanisms. Apex code runs in atomic transactions.",
    ),
    (
        "Debugging, Testing, and Deploying Apex",
        677,
        791,
        "Develop your Apex code in a sandbox and debug it with the Developer "
        "Console and debug logs. Unit-test your code, then distribute it to "
        "customers using packages.",
    ),
    (
        "Apex Reference",
        792,
        792,
        "In Summer '21 and later versions, Apex reference content is moved to "
        "a separate guide called the Apex Reference Guide.",
    ),
    (
        "Appendices",
        793,
        821,
        "Supplementary reference material: versioned behavior changes, a "
        "full worked example, reserved keywords, and documentation "
        "typographical conventions.",
    ),
]

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Hybrid retrieval
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6
RETRIEVAL_TOP_K = 5
SECTION_BOOST_WEIGHT = 0.15  # soft boost, not a hard filter

# Guardrail threshold, tuned empirically during verification: on real Apex
# questions this scored 0.39-0.48; off-topic questions scored -0.11-0.23.
# (Note: the section router's own confidence score turned out not to
# discriminate on-topic from off-topic questions at all -- e.g. "what's the
# weather today" scored higher than a real cross-section Apex question -- so
# it's used only for the soft re-ranking boost above, not as a guardrail.)
RETRIEVAL_SCORE_THRESHOLD = 0.3
OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions about the Apex Developer Guide. "
    "That question doesn't seem to be covered by it."
)


def section_for_page(page_number: int) -> str:
    for name, first, last, _ in TOP_LEVEL_SECTIONS:
        if first <= page_number <= last:
            return name
    return "Unknown"
