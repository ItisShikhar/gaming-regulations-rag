# Compliance RAG Learning Bench

This repository is a learning bench for Python, retrieval-augmented generation, and search evaluation.

It indexes public gaming rules and technical standards from New Jersey, the United Kingdom, and GLI.

It does not generate answers with a language model. It retrieves source chunks and reports their citations.

## What this project does

The pipeline downloads source files, extracts text, creates section-aware chunks, and builds local search indexes.

The search pipeline combines dense vectors with BM25 keyword search. Reciprocal Rank Fusion combines the two ranked lists.

Each chunk stores its jurisdiction, document type, section, clause path, source file, source URL, and page range.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/download_corpus.py` | Download the PDF and UKGC web sources. |
| `src/extract_pdfs.py` | Extract searchable text from the PDF files. |
| `src/ingest.py` | Create character-bounded chunks with section and clause metadata. |
| `src/index.py` | Split chunks by model tokens and build the Qdrant and BM25 indexes. |
| `src/search.py` | Search chunks with dense, BM25, hybrid, or jurisdiction-routed retrieval. |
| `eval/compliance_questions.jsonl` | Store questions and expected source sections. |
| `eval/run_eval.py` | Measure retrieval quality against the expected sections. |
| `eval/test_compliance.py` | Test chunk parsing and retrieval behavior. |

The `data/raw`, `data/processed`, `data/qdrant-compliance`, and `results` directories are local. Git ignores them.

## Requirements

Use Python 3.13 for the pinned package versions in this project.

Install the packages from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Build the local corpus

Run these commands from the repository root, in order:

```powershell
.\.venv\Scripts\python.exe src\download_corpus.py
.\.venv\Scripts\python.exe src\extract_pdfs.py
.\.venv\Scripts\python.exe src\ingest.py
.\.venv\Scripts\python.exe src\index.py
```

The downloader saves source files under `data/raw/compliance`.

The PDF extractor writes Markdown beside each PDF file. The ingester writes `data/processed/compliance_chunks.jsonl`.

The indexer creates token-bounded chunks at `data/processed/compliance_index_chunks.jsonl`.
It creates the `compliance` Qdrant collection under `data/qdrant-compliance`.
It also writes `data/processed/compliance_bm25.pkl` and an index manifest.

The indexer replaces the local `compliance` collection when it builds a new index.
Close any search process before you rebuild the index.

The manifest stores the corpus fingerprint, pinned embedding model revision, tokenizer version, and point count.
Search stops with an error when the manifest does not match the current chunks.

Local Qdrant allows one process to use its store at a time.
Do not run search and evaluation commands in parallel.

## Search

Search with hybrid retrieval:

```powershell
.\.venv\Scripts\python.exe src\search.py "What does N.J.A.C. 13:69O-1.2(e)(1) require about mobile gaming location?"
```

Compare pre-filtering, post-filtering, and multi-query routing:

```powershell
.\.venv\Scripts\python.exe src\search.py "Compare UKGC RTS requirement 7A and GLI-19 section 3.2.1" --mode compare
```

Filter results to one jurisdiction or regulation type:

```powershell
.\.venv\Scripts\python.exe src\search.py "random number generation" --jurisdiction ukgc
.\.venv\Scripts\python.exe src\search.py "account controls" --regulation-type administrative_code
```

The search modes work as follows:

- **Dense:** ranks chunks by vector similarity. It can match a concept when the question uses different words.
- **BM25:** ranks chunks by token overlap. The tokenizer keeps legal and technical identifiers for exact searches.
- **Hybrid:** combines dense and BM25 ranks with RRF. Exact references receive a promotion when the question names a recognized section.
- **Pre-filter:** applies metadata filters before each retriever selects its candidates.
- **Post-filter:** selects global candidates first, then removes chunks outside the filters. This mode can return fewer relevant chunks.
- **Multi-query:** searches the same question once per selected jurisdiction, then merges the results. It does not rewrite the question.

BM25 uses corpus-wide token statistics in both filter modes. A separate BM25 index for each jurisdiction is a possible next experiment.

## How chunking works

The chunker uses structure in each source file. It does not ask a model to guess where a thought ends.

- UKGC Markdown headings define section boundaries.
- NJ rules use markers such as `13:69O-1.2`, `(a)`, `1.`, and `i.`.
- GLI-19 uses chapter names and numbered sections such as `3.2.1`.
- Blank lines split paragraphs. PDF page markers add page metadata but do not end a section.
- The ingestion step packs neighboring blocks up to 1,000 characters.
- Long blocks split at sentence or semicolon boundaries. The ingestion step splits long words only when no earlier boundary fits.
- The index step uses the model tokenizer to keep each indexed chunk at or below 220 tokens.
- The model accepts up to 256 tokens. The 220-token limit includes special tokens and leaves 36 spare tokens.

Each new heading or numbered section starts a new section.
The stored heading path gives the section's parent context.

This is a structural heuristic.
A source with missing headings or poor PDF extraction can produce poor chunks.
Inspect sample chunks before you trust search results.

## Evaluate retrieval

Run the compliance benchmark:

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py
```

The evaluation reports Recall@5 and Mean Reciprocal Rank. It checks every expected section against the generated chunks before it scores retrieval.

The current evaluation file has four questions. Its scores are a smoke test, not evidence of production quality.

Add questions that test exact citations, paraphrases, cross-jurisdiction comparisons, and questions that the corpus cannot answer. Keep expected section labels tied to source text.

Run the unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest eval.test_compliance -v
```

Compile the Python files:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\download_corpus.py src\extract_pdfs.py src\ingest.py src\index.py src\search.py eval\run_eval.py eval\test_compliance.py
```

## Source and legal-use notes

The UKGC pages come from the [Remote Gambling and Software Technical Standards](https://www.gamblingcommission.gov.uk/standards/remote-gambling-and-software-technical-standards) site.

The New Jersey PDFs come from the [Division of Gaming Enforcement regulations page](https://www.nj.gov/oag/ge/regulations.html).

The [official GLI standards page](https://gaminglabs.com/gli-standards/) lists the published GLI standards.
The GLI-19 PDF comes from a third-party GitHub mirror.
Direct access to GLI's file was blocked during collection.

The collected files are snapshots.
The New Jersey PDFs date from 2016 and 2017.
They can be older than the current rules.
Check each source and its effective date before you use a result for a compliance decision.

The chunks include a source URL and PDF page numbers.
They do not include verified effective dates or amendment histories.

This project is for learning and research. It is not legal advice and does not determine whether a product complies with a rule.

The corpus and generated indexes stay out of Git because source documents can have separate copyright and reuse terms.
