"""CLI: ``python -m grounded_rag.ask "your question"``. Prints the grounded answer."""

from __future__ import annotations

import argparse

from .answer import get_answer_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the grounded RAG system a question.")
    parser.add_argument("question", type=str)
    parser.add_argument("--json", action="store_true", help="emit the full structured answer")
    args = parser.parse_args()

    service = get_answer_service()
    result = service.ask(args.question)

    if args.json:
        print(result.model_dump_json(indent=2))
        return

    print(f"\nQ: {args.question}\n")
    print(result.answer)
    if result.citations:
        print("\ncitations:")
        for c in result.citations:
            print(f"  - {c.label}")
    print(f"\n[refused={result.refused}, contexts={len(result.contexts)}]")


if __name__ == "__main__":
    main()
