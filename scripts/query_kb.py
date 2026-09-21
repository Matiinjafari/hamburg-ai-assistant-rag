"""Answer questions using the indexed community knowledge base."""

import argparse
import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langdetect import LangDetectException, detect
from openai import OpenAI
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
CHAT_MODEL = "gpt-4o-mini"

TOP_K = 6
MIN_SIMILARITY = 0.35

SYSTEM_PROMPT = """
Du beantwortest praktische Fragen für eine Telegram-Gruppe iranischer
Studierender in Hamburg. Verwende ausschließlich die bereitgestellten Einträge
aus der Wissensbasis und ergänze keine Informationen aus eigenem Wissen.

Antworte in derselben Sprache wie die Frage. Die Sprache der Wissensbasis oder
dieser Anweisung darf die Antwortsprache nicht beeinflussen.

Weitere Regeln:

- Wenn sich Informationen aus verschiedenen Zeiträumen unterscheiden, erkläre
  die zeitliche Entwicklung.
- Gib Hinweise auf unsichere oder möglicherweise veraltete Informationen weiter.
- Wenn die Einträge die Frage nicht zuverlässig beantworten, sage das ehrlich.
- Halte die Antwort kurz, direkt und praktisch.
""".strip()

NO_RESULT_MESSAGES = {
    "Persisch (Farsi)": (
        "در گروه اطلاعات قابل اعتمادی درباره این موضوع پیدا نشد."
    ),
    "Deutsch": (
        "Dazu wurde in der Gruppe nichts Verlässliches gefunden."
    ),
    "Englisch": (
        "No reliable information about this was found in the group."
    ),
}


def load_index(index_dir: str):
    index_path = Path(index_dir)

    index = faiss.read_index(
        str(index_path / "index.faiss")
    )

    with open(
        index_path / "entries.json",
        encoding="utf-8",
    ) as file:
        entries = json.load(file)

    return index, entries


def retrieve(
    model,
    index,
    entries,
    question: str,
    limit: int = TOP_K,
):
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )
    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    result_count = min(limit, index.ntotal)
    if result_count == 0:
        return []

    scores, positions = index.search(
        query_embedding,
        result_count,
    )

    results = []

    for score, position in zip(
        scores[0],
        positions[0],
    ):
        if position == -1 or score < MIN_SIMILARITY:
            continue

        results.append(
            (float(score), entries[position])
        )

    return results


def build_context(results) -> str:
    blocks = []

    for score, entry in results:
        block = (
            f"[Ähnlichkeit: {score:.2f} | "
            f"Zeitraum: {entry.get('era_label', '')} | "
            f"Kategorie: {entry.get('category', '')}]\n"
            f"Thema: {entry.get('topic', '')}\n"
            f"Frage: {entry['question']}\n"
            f"Antwort: {entry['answer']}"
        )

        if entry.get("note"):
            block += f"\nHinweis: {entry['note']}"

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def detect_language_name(text: str) -> str | None:
    # Short colloquial Farsi is often misclassified as Urdu.
    if re.search(r"[\u0600-\u06FF]", text):
        return "Persisch (Farsi)"

    try:
        language_code = detect(text)
    except LangDetectException:
        return None

    language_names = {
        "de": "Deutsch",
        "en": "Englisch",
    }
    return language_names.get(
        language_code,
        language_code,
    )


def answer_question(
    client: OpenAI,
    model,
    index,
    entries,
    question: str,
):
    language = detect_language_name(question)
    results = retrieve(
        model,
        index,
        entries,
        question,
    )

    if not results:
        message = NO_RESULT_MESSAGES.get(
            language,
            NO_RESULT_MESSAGES["Englisch"],
        )
        return message, []

    context = build_context(results)

    if language:
        language_instruction = (
            f"Antworte auf {language}."
        )
    else:
        language_instruction = (
            "Antworte in der Sprache der Frage."
        )

    user_prompt = (
        f"WISSENSBASIS:\n\n{context}\n\n"
        f"FRAGE:\n{question}\n\n"
        f"{language_instruction}"
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.choices[0].message.content, results


def print_answer(
    client,
    model,
    index,
    entries,
    question,
):
    answer, results = answer_question(
        client,
        model,
        index,
        entries,
        question,
    )

    print(f"\nAntwort:\n{answer}\n")

    if results:
        print("Quellen:")

        for score, entry in results:
            topic = entry.get("topic", "")
            era = entry.get("era_label", "")

            print(
                f"  - [{score:.2f}] {topic} ({era})"
            )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions using the knowledge base."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="question to ask; omit for interactive mode",
    )
    parser.add_argument(
        "--index-dir",
        default="kb_index",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to .env "
            "or export it before running this script."
        )

    client = OpenAI(api_key=api_key)

    print(f"Loading model and index from {args.index_dir}")

    model = SentenceTransformer(MODEL_NAME)
    index, entries = load_index(args.index_dir)

    print("Ready")

    if args.question:
        print_answer(
            client,
            model,
            index,
            entries,
            args.question,
        )
        return

    print(
        "Interactive mode. Enter a question or "
        "'exit' to stop."
    )

    while True:
        question = input("> ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if question:
            print_answer(
                client,
                model,
                index,
                entries,
                question,
            )


if __name__ == "__main__":
    main()
