# 📚 RAG Document Intelligence System

Upload a collection of documents (PDF, DOCX, TXT, MD), ask complex questions, and get answers
**grounded in your documents, with citations** to the exact file, page, and passage.

Built entirely with free tools. It runs locally and uses Google's free Gemini tier for generation.

<!-- Add a demo GIF here: ![demo](docs/demo.gif) -->

## Highlights

- **Hybrid retrieval**: dense vectors + BM25 keyword search, merged with Reciprocal Rank Fusion,
  then re-scored by a cross-encoder reranker.
- **Measured, not assumed**: on a 28-question benchmark over 1,200+ pages, recall@5 improved
  from **0.57 (vector only) to 0.93 (hybrid + rerank)**.
- **Cited answers**: every claim carries a `[n]` marker linking to a source passage (file + page).
  Cited passages are shown first; other retrieved passages are collapsed.
- **Refuses to guess**: if the documents don't contain the answer it says so, and when only part
  of a question can be answered, it answers that part and states what is missing.
- **Multi-part questions**: comparisons are broken into sub-queries and the evidence is merged.
- **Conversation memory**: follow-ups like "how does that compare with last year?" are rewritten
  into standalone questions before retrieval.

## How it works

```
Upload → parse → sentence-aware chunking (page metadata) → embed → ChromaDB
                                                                  │
Question → (rewrite if follow-up) → (decompose if multi-part) ────┤
                                                                  ▼
          Vector search (MiniLM)  ┐
                                  ├→ Reciprocal Rank Fusion → Cross-encoder rerank → top-k
          BM25 keyword search     ┘                                                    │
                                                                                       ▼
                                  LLM answers ONLY from the passages, cites [n], or declines
```

| Component | Choice |
|---|---|
| UI | Streamlit |
| Parsing | PyMuPDF, python-docx |
| Embeddings | `all-MiniLM-L6-v2` (local) |
| Vector store | ChromaDB (persistent, local) |
| Keyword search | `rank_bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| LLM | Gemini free tier (swappable: only `src/llm.py` talks to the provider) |

## Evaluation

**Corpus:** India's Economic Survey 2024-25 (482 pages) and 2025-26 (740 pages), about
1,200 pages and 5,000 chunks.

**Benchmark:** 28 hand-written questions (`eval/questions.json`), each with a source document and
an evidence phrase that must appear in a retrieved chunk. Questions are paraphrased rather than
copied from the text, and retrieval is *not* restricted to the right document, so it must also
tell the two years apart. Metric: recall@k (no LLM calls needed).

| Retrieval mode | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|
| Vector only | 0.29 | 0.57 | 0.57 |
| Hybrid (vector + BM25, RRF) | 0.39 | 0.68 | 0.75 |
| Hybrid + cross-encoder rerank | **0.61** | **0.82** | **0.93** |

What the numbers show: keyword search helps because the questions hinge on exact terms like
`FY26`, `H1`, and specific figures that embeddings blur. Reranking helps most at the top
of the list, since the cross-encoder reads question and passage together and can tell FY25 from
FY26.

### Generation checks (manual)

Retrieval metrics don't prove the final answers are right, so I also checked 11 questions by hand
against the cited passages: cross-document comparisons, a follow-up question, refusals, and a
document-filter test (`eval/manual_test_questions.md`). Testing found three real problems, all
fixed:

1. An answer implied a 38.61 GW *addition* was a running total. The prompt now requires periods on
   figures and says when a number is a change.
2. A follow-up ("how does that compare…?") was answered against the wrong comparison. Answering now
   uses the rewritten standalone question.
3. A filtered query refused outright although part of the answer existed. Partial answers are now
   allowed.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # add a free key from https://aistudio.google.com/apikey
streamlit run app.py
```

The first run downloads two small models (about 150 MB). Gemini model names change often. If you
get a 404 or timeout, run `python check_models.py` to find one that works for your key and set
`GEMINI_MODEL` in `.env`.

### Reproduce the evaluation

```bash
# Put the two Survey PDFs in a folder named exactly:
#   economic_survey_2025-26.pdf   economic_survey_2024-25.pdf
python -m eval.evaluate --docs path/to/folder --questions eval/questions.json
```

## Design decisions

- **Few LLM calls.** The free tier is rate-limited, so local heuristics decide when a question
  needs rewriting or decomposition. A simple lookup costs one call.
- **Fail fast.** Quota, key, and model errors stop immediately with a plain-English message;
  only transient server errors get one retry.
- **Swappable LLM.** The provider is isolated in `src/llm.py`.
- **Sentence-aware chunking with overlap** so facts are rarely cut in half, with page numbers kept
  on every chunk for citations.

## Limitations

- Scanned PDFs need OCR (not included). Tables and charts are extracted as plain text, so
  chart-only numbers can be misread.
- 28 benchmark questions is small (one question is about 3.6 points), and most are numeric lookups.
  Evidence matching is strict, so true recall is probably slightly higher.
- Answer faithfulness was checked manually on 11 questions, not scored automatically.
- The decomposition and follow-up heuristics are simple keyword rules and will occasionally misjudge.
- BM25 is rebuilt in memory after each upload: fine for thousands of chunks, not millions.
- A small free model can still phrase things awkwardly. The citations are what make errors checkable.

## Project structure

```
app.py                  Streamlit UI
check_models.py         Find which Gemini models work for your key
src/config.py           Tunable settings (chunk size, top-k, models)
src/ingestion.py        Parsing + chunking
src/store.py            ChromaDB + BM25
src/retrieval.py        RRF + reranking
src/pipeline.py         Query rewriting, decomposition, prompting, citations
src/llm.py              LLM wrapper with error handling
eval/evaluate.py        Recall@k benchmark
eval/questions.json     28 benchmark questions
eval/manual_test_questions.md   Hand-check list (cross-doc, refusals, follow-ups)
```

## Deploy (Streamlit Community Cloud)

1. Push to GitHub (`.env`, `data/`, and PDFs are git-ignored).
2. Create the app from `app.py` on share.streamlit.io.
3. Add `GEMINI_API_KEY` under **Settings → Secrets**.

The free tier has limited memory and an ephemeral disk, so a demo works best with a small
document set uploaded at runtime. A shared demo also shares one free Gemini quota.
