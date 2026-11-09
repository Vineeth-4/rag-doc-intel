"""Retrieval evaluation: recall@k for vector vs hybrid vs hybrid+rerank. No LLM calls needed.

Usage (from project root):
    python -m eval.evaluate --docs path/to/docs_folder --questions eval/questions.json

questions.json format:
    [{"question": "...", "source": "paper.pdf", "evidence": "a short phrase that appears
      in the passage that answers the question"}]

A retrieval counts as a hit if any of the top-k chunks (from the right file) contains the
evidence phrase (case- and whitespace-insensitive).
"""
import argparse
import json
from pathlib import Path

from src.config import CHROMA_DIR
from src.ingestion import SUPPORTED_EXTENSIONS, ingest_file
from src.retrieval import MODES, Retriever
from src.store import DocumentStore


def _norm(text: str) -> str:
    """Lowercase and collapse all whitespace so PDF line-break quirks never cause misses."""
    return " ".join(text.lower().split())


def recall_at_k(retriever: Retriever, questions: list[dict], mode: str, k: int) -> float:
    hits = 0
    for q in questions:
        results = retriever.retrieve(q["question"], mode=mode, top_k=k)
        evidence = _norm(q["evidence"])
        if any(r["source"] == q["source"] and evidence in _norm(r["text"]) for r in results):
            hits += 1
    return hits / len(questions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", help="Folder of documents to index first (optional)")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    # Separate directory so evaluation never touches your app's index.
    store = DocumentStore(persist_dir=CHROMA_DIR.parent / "chroma_eval")
    if args.docs:
        for path in sorted(Path(args.docs).iterdir()):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                chunks = ingest_file(path.name, path.read_bytes())
                store.add_chunks(chunks)
                print(f"Indexed {path.name}: {len(chunks)} chunks")

    questions = json.loads(Path(args.questions).read_text())
    retriever = Retriever(store)

    header = "| Mode | " + " | ".join(f"Recall@{k}" for k in args.ks) + " |"
    print(f"\n{len(questions)} questions\n\n{header}\n|---|" + "---|" * len(args.ks))
    for mode in MODES:
        row = [f"{recall_at_k(retriever, questions, mode, k):.2f}" for k in args.ks]
        print(f"| {mode} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
