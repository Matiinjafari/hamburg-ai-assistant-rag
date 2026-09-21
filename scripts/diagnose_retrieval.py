"""Inspect similarity rankings for a knowledge-base query."""

import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

from query_kb import MODEL_NAME, load_index


def main():
    parser = argparse.ArgumentParser(
        description="Show retrieval scores for a question."
    )
    parser.add_argument("question")
    parser.add_argument(
        "--index-dir",
        default="kb_index",
    )
    parser.add_argument(
        "--highlight",
        help="ID of an entry to highlight",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="number of results to display",
    )
    args = parser.parse_args()

    print(f"Loading index from {args.index_dir}")

    model = SentenceTransformer(MODEL_NAME)
    index, entries = load_index(args.index_dir)

    if index.ntotal == 0:
        raise SystemExit("The index is empty")

    query_embedding = model.encode(
        [args.question],
        normalize_embeddings=True,
    )
    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, positions = index.search(
        query_embedding,
        index.ntotal,
    )

    print(f"\nQuery: {args.question}")
    print(
        f"Showing the top {min(args.top, index.ntotal)} "
        f"of {index.ntotal} results:\n"
    )

    highlighted_rank = None

    for rank, (score, position) in enumerate(
        zip(scores[0], positions[0]),
        start=1,
    ):
        entry = entries[position]
        is_highlighted = (
            args.highlight
            and entry["id"] == args.highlight
        )

        if is_highlighted:
            highlighted_rank = rank

        if rank <= args.top or is_highlighted:
            marker = "  << highlighted" if is_highlighted else ""
            topic = entry.get("topic", "")[:70]

            print(
                f"  #{rank:3d}  [{score:.3f}]  "
                f"{entry['id']:6s}  {topic}{marker}"
            )

    if args.highlight and highlighted_rank is None:
        print(
            f"\nEntry '{args.highlight}' was not found "
            "in the index."
        )
    elif args.highlight:
        print(
            f"\nEntry '{args.highlight}' ranked "
            f"#{highlighted_rank} of {index.ntotal}."
        )


if __name__ == "__main__":
    main()
