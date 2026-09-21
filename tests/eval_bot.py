"""Evaluate answer accuracy and refusal behavior."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from query_kb import (  # noqa: E402
    MODEL_NAME,
    answer_question,
    load_index,
)


load_dotenv()

JUDGE_MODEL = "gpt-4o-mini"

JUDGE_PROMPT = """
Vergleiche die Bot-Antwort mit der Referenzantwort. Bewerte die inhaltliche
Übereinstimmung, nicht den Wortlaut.

Setze "consistent" auf false, wenn die Bot-Antwort wesentliche Informationen
auslässt, der Referenz widerspricht oder nicht belegte Fakten ergänzt.

Antworte ausschließlich als JSON:

{
  "consistent": true,
  "reason": "Kurze Begründung auf Deutsch"
}
""".strip()

TRICK_QUESTIONS = [
    "Wie repariere ich mein Fahrrad?",
    "چطور می‌تونم یک کیک شکلاتی بپزم؟",
    "What's the best programming language to learn in 2026?",
    "چه فیلم‌هایی رو توصیه می‌کنی ببینم؟",
    "Wie trainiere ich meinen Hund?",
]

REFUSAL_REFERENCE = (
    "Der Bot sollte ehrlich sagen, dass er dazu keine "
    "verlässlichen Informationen hat, statt eine Antwort "
    "zu erfinden."
)


def judge(
    client: OpenAI,
    question: str,
    reference_answer: str,
    bot_answer: str,
) -> dict:
    prompt = (
        f"FRAGE:\n{question}\n\n"
        f"REFERENZANTWORT:\n{reference_answer}\n\n"
        f"BOT-ANTWORT:\n{bot_answer}"
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": JUDGE_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return json.loads(
        response.choices[0].message.content
    )


def build_sample(
    categorized_path: str,
    entries_per_category: int,
    seed: int,
):
    with open(
        categorized_path,
        encoding="utf-8",
    ) as file:
        entries_by_category = json.load(file)

    random_generator = random.Random(seed)
    sample = []

    for entries in entries_by_category.values():
        answered_entries = [
            entry
            for entry in entries
            if entry.get("answer_type") != "unanswered"
            and entry.get("answer")
        ]

        sample_size = min(
            entries_per_category,
            len(answered_entries),
        )
        sample.extend(
            random_generator.sample(
                answered_entries,
                sample_size,
            )
        )

    return sample


def evaluate_known_answers(
    client,
    embedding_model,
    index,
    indexed_entries,
    sample,
):
    results = []
    consistent_count = 0

    print(
        f"\nKnown-answer evaluation "
        f"({len(sample)} questions)\n"
    )

    for position, entry in enumerate(
        sample,
        start=1,
    ):
        print(
            f"[{position}/{len(sample)}] "
            f"{entry['topic'][:60]}",
            end=" ",
            flush=True,
        )

        bot_answer, _ = answer_question(
            client,
            embedding_model,
            index,
            indexed_entries,
            entry["question"],
        )

        verdict = judge(
            client,
            entry["question"],
            entry["answer"],
            bot_answer,
        )

        result = {
            "id": entry["id"],
            "category": entry.get("category", ""),
            "question": entry["question"],
            "reference_answer": entry["answer"],
            "bot_answer": bot_answer,
            "consistent": verdict["consistent"],
            "reason": verdict["reason"],
        }
        results.append(result)

        if verdict["consistent"]:
            consistent_count += 1
            print("OK")
        else:
            print(
                f"Mismatch: {verdict['reason'][:80]}"
            )

    accuracy = (
        consistent_count / len(sample)
        if sample
        else 0
    )

    print(
        f"\nAccuracy: {consistent_count}/{len(sample)} "
        f"({accuracy:.0%})"
    )

    return results, consistent_count, accuracy


def evaluate_refusals(
    client,
    embedding_model,
    index,
    indexed_entries,
    questions,
):
    results = []
    refusal_count = 0

    print(
        f"\nRefusal evaluation "
        f"({len(questions)} questions)\n"
    )

    for position, question in enumerate(
        questions,
        start=1,
    ):
        print(
            f"[{position}/{len(questions)}] "
            f"{question[:50]}",
            end=" ",
            flush=True,
        )

        bot_answer, retrieved = answer_question(
            client,
            embedding_model,
            index,
            indexed_entries,
            question,
        )

        verdict = judge(
            client,
            question,
            REFUSAL_REFERENCE,
            bot_answer,
        )
        correctly_refused = verdict["consistent"]

        results.append(
            {
                "question": question,
                "bot_answer": bot_answer,
                "n_retrieved": len(retrieved),
                "correctly_refused": correctly_refused,
                "reason": verdict["reason"],
            }
        )

        if correctly_refused:
            refusal_count += 1
            print("OK")
        else:
            print("Failed")

    refusal_rate = (
        refusal_count / len(questions)
        if questions
        else 0
    )

    print(
        f"\nCorrect refusals: "
        f"{refusal_count}/{len(questions)} "
        f"({refusal_rate:.0%})"
    )

    return results, refusal_count, refusal_rate


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG question-answering pipeline."
    )
    parser.add_argument(
        "--categorized",
        default="knowledge_base/qa_categorized.json",
    )
    parser.add_argument(
        "--index-dir",
        default="kb_index",
    )
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--n-trick",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "-o",
        "--output",
        default="eval_results.json",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    print("Loading embedding model and index")

    embedding_model = SentenceTransformer(MODEL_NAME)
    index, indexed_entries = load_index(
        args.index_dir
    )

    sample = build_sample(
        args.categorized,
        args.n_per_category,
        args.seed,
    )

    known_results, consistent_count, accuracy = (
        evaluate_known_answers(
            client,
            embedding_model,
            index,
            indexed_entries,
            sample,
        )
    )

    trick_questions = TRICK_QUESTIONS[:args.n_trick]

    refusal_results, refusal_count, refusal_rate = (
        evaluate_refusals(
            client,
            embedding_model,
            index,
            indexed_entries,
            trick_questions,
        )
    )

    results = {
        "known_answer": known_results,
        "trick_questions": refusal_results,
        "summary": {
            "known_answer_accuracy": accuracy,
            "known_answer_n": len(sample),
            "refusal_rate": refusal_rate,
            "refusal_n": len(trick_questions),
        },
    }

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nResults written to {args.output}")
    print("\nSummary")
    print(
        f"Known-answer accuracy: "
        f"{accuracy:.0%} "
        f"({consistent_count}/{len(sample)})"
    )
    print(
        f"Correct refusal rate: "
        f"{refusal_rate:.0%} "
        f"({refusal_count}/{len(trick_questions)})"
    )


if __name__ == "__main__":
    main()
