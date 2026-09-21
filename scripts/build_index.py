"""Build a FAISS index from the Q&A knowledge base."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


def main():
    parser = argparse.ArgumentParser(
        description="Build a multilingual search index."
    )
    parser.add_argument(
        "qa_file",
        help="JSON file containing the Q&A entries",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default="kb_index",
    )
    args = parser.parse_args()

    with open(args.qa_file, encoding="utf-8") as file:
        entries = json.load(file)

    entries = [
        entry
        for entry in entries
        if entry.get("answer_type") != "unanswered"
        and entry.get("answer")
    ]

    if not entries:
        raise SystemExit("No answered entries found")

    print(f"Indexing {len(entries)} entries")

    model = SentenceTransformer(MODEL_NAME)

    # Including the topic helps with short or ambiguous questions.
    texts = [
        f"{entry.get('topic', '')}. {entry['question']}"
        for entry in entries
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    # Inner product on normalized vectors is cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(
        index,
        str(output_dir / "index.faiss"),
    )

    with open(
        output_dir / "entries.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            entries,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Index and {len(entries)} entries "
        f"written to {output_dir}"
    )


if __name__ == "__main__":
    main()
