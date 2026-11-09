"""Question -> (condense) -> (decompose) -> retrieve -> grounded, cited answer."""
import json
import re

from . import llm
from .config import MAX_CONTEXT_CHUNKS, TOP_K_FINAL
from .retrieval import Retriever

NOT_FOUND = "I couldn't find this in the uploaded documents."

ANSWER_PROMPT = """You are a document question-answering assistant.
Answer the question using ONLY the numbered context passages below.

Rules:
- Cite every claim with the passage number in square brackets, e.g. [1] or [2][3].
- Give the time period for every figure. When a figure is an increase, addition or change
  (rather than an overall level), say so explicitly. Use plain wording for ordinary levels:
  never describe a figure as a "total level". Do not present figures from different periods
  as a sequence unless the passages themselves do.
- If the passages answer only part of the question, answer that part and say clearly what is
  missing. Use the exact sentence "{not_found}" only when nothing relevant is found.
- Do not use outside knowledge. Do not invent citations.
- Be concise. For comparisons or multi-part questions, address every part.

Context passages:
{context}

Question: {question}

Answer:"""

DECOMPOSE_PROMPT = """Break the question into the minimal set of simple, self-contained search
queries needed to answer it. If it is already simple, return just one query.
Return ONLY a JSON array of strings, with at most 4 items.

Question: {question}"""

CONDENSE_PROMPT = """Given the chat history and a follow-up question, rewrite the follow-up as a
standalone question. Return ONLY the rewritten question.

History:
{history}

Follow-up: {question}"""


# Cheap local checks so we only spend an LLM call when it is likely to help.
_COMPLEX_RX = re.compile(
    r"\b(compare|comparison|versus|vs|difference|differ|differs|contrast|both|"
    r"as well as|respectively|summari[sz]e|each of)\b", re.I)
_FOLLOWUP_RX = re.compile(
    r"\b(it|its|that|this|those|these|they|them|their|he|she|former|latter|same|"
    r"also|previous|earlier|above)\b", re.I)


def looks_complex(question: str) -> bool:
    return len(question.split()) > 18 or question.count("?") > 1 or bool(_COMPLEX_RX.search(question))


def looks_like_followup(question: str) -> bool:
    return len(question.split()) <= 6 or bool(_FOLLOWUP_RX.search(question))


def _format_history(history: list[dict], n: int = 6) -> str:
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-n:])


def condense_question(question: str, history: list[dict]) -> str:
    if not history or not looks_like_followup(question):
        return question  # already standalone: skip the LLM call
    out = llm.generate(CONDENSE_PROMPT.format(history=_format_history(history), question=question))
    return out or question


def decompose_question(question: str) -> list[str]:
    raw = llm.generate(DECOMPOSE_PROMPT.format(question=question), temperature=0.0)
    try:  # only a malformed reply falls back; LLM failures propagate so we stop fast
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        subs = json.loads(match.group(0)) if match else []
        subs = [s for s in subs if isinstance(s, str) and s.strip()][:4]
        return subs or [question]
    except (ValueError, TypeError):
        return [question]


def answer_question(retriever: Retriever, question: str, sources: list[str] | None = None,
                    history: list[dict] | None = None, use_decomposition: bool = True,
                    mode: str = "hybrid_rerank") -> dict:
    history = history or []
    standalone = condense_question(question, history)
    sub_questions = (decompose_question(standalone)
                     if use_decomposition and looks_complex(standalone) else [standalone])

    # Retrieve for each sub-question, dedupe by chunk id keeping the best score.
    merged: dict[str, dict] = {}
    for sq in sub_questions:
        for hit in retriever.retrieve(sq, sources, mode=mode, top_k=TOP_K_FINAL):
            if hit["id"] not in merged or hit["score"] > merged[hit["id"]]["score"]:
                merged[hit["id"]] = hit
    hits = sorted(merged.values(), key=lambda h: h["score"], reverse=True)[:MAX_CONTEXT_CHUNKS]

    if not hits:
        return {"answer": NOT_FOUND, "sources": [], "sub_questions": sub_questions}

    context = "\n\n".join(
        f"[{i}] (file: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )
    # Use the standalone (rewritten) question so follow-ups like "how does that compare...?"
    # are answered against what "that" actually refers to.
    answer = llm.generate(ANSWER_PROMPT.format(
        not_found=NOT_FOUND, context=context, question=standalone))
    if answer.strip().strip('"') == NOT_FOUND:
        hits = []  # a refusal should not display unrelated "sources"
    return {"answer": answer, "sources": hits, "sub_questions": sub_questions}
