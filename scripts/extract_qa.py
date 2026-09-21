"""Extract structured Q&A pairs from Telegram conversation clusters."""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """
Du extrahierst praktische Frage-Antwort-Paare aus Ausschnitten einer
mehrsprachigen Telegram-Gruppe iranischer Studierender in Hamburg. Die
Nachrichten sind überwiegend auf Farsi, teilweise auf Deutsch oder Englisch.

Berücksichtige praktische Themen wie Aufenthalt und Visum, Pass und Botschaft,
Banken, Krankenversicherung, Wohnen, Studium, Arbeit, Führerschein und Alltag.

Regeln:

- Entferne die Namen der Absender. Namen von Institutionen, Ärzten, Produkten
  oder Behörden dürfen erhalten bleiben.
- Eine Antwort kann über die Telegram-Reply-Funktion verknüpft sein oder sich
  nur aus dem Gesprächskontext ergeben.
- Verwende "explicit_reply" für direkte Antworten und "implicit_context" für
  Antworten, die sich aus späteren Nachrichten ergeben.
- Ignoriere Grüße, Witze, Emojis, Anzeigen sowie politische oder aktivistische
  Diskussionen ohne sachliche Frage und Antwort.
- Nimm praktische Fragen ohne Antwort mit "answer": null und
  "answer_type": "unanswered" auf.
- Erfinde keine Informationen. Kennzeichne Vermutungen, Gerüchte und
  möglicherweise veraltete Angaben im Feld "note".
- Behalte die Originalsprache der Frage und Antwort bei.

Gib ausschließlich ein JSON-Objekt mit dem Feld "qa_pairs" zurück. Jeder
Eintrag kann folgende Felder enthalten:

- "topic": kurzes Thema auf Deutsch
- "question_lang": "fa", "en" oder "de"
- "question": Frage in der Originalsprache
- "answer": Antwort in der Originalsprache oder null
- "answer_type": "explicit_reply", "implicit_context" oder "unanswered"
- "note": Hinweis auf Unsicherheit oder veraltete Informationen
- "source_msg_ids": IDs der verwendeten Nachrichten als Strings

Wenn der Cluster keine geeigneten Inhalte enthält, gib {"qa_pairs": []} zurück.
""".strip()


def format_cluster(cluster: dict) -> str:
    lines = [
        (
            f"Cluster {cluster['cluster_id']} "
            f"({cluster['start']} - {cluster['end']}):"
        )
    ]

    for message in cluster["messages"]:
        reply = (
            f" [reply_to={message['reply_to']}]"
            if message.get("reply_to")
            else ""
        )
        media = (
            f" [MEDIA:{message['media_type']}]"
            if message.get("media_type")
            else ""
        )
        text = (
            message.get("text") or ""
        ).replace("\n", " | ")

        lines.append(
            f"[{message['id']}]{reply}{media}: {text}"
        )

    return "\n".join(lines)


def extract_from_cluster(
    client: OpenAI,
    cluster: dict,
    retries: int = 3,
) -> list:
    if cluster["size"] < 2:
        return []

    content = format_cluster(cluster)

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"},
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
            )

            result = json.loads(
                response.choices[0].message.content
            )
            return result.get("qa_pairs", [])
        except Exception as error:
            print(
                f"Warning: cluster {cluster['cluster_id']}, "
                f"attempt {attempt}/{retries}: {error}",
                file=sys.stderr,
            )

            if attempt < retries:
                time.sleep(2 ** (attempt - 1))

    print(
        f"Cluster {cluster['cluster_id']} failed "
        f"after {retries} attempts",
        file=sys.stderr,
    )
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract Q&A pairs from Telegram clusters."
    )
    parser.add_argument(
        "clusters_file",
        help="JSONL file produced by build_clusters.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="qa_dataset_extracted.json",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="first number used for generated Q&A IDs",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to .env "
            "or export it before running this script."
        )

    client = OpenAI(api_key=api_key)

    with open(
        args.clusters_file,
        encoding="utf-8",
    ) as file:
        clusters = [
            json.loads(line)
            for line in file
        ]

    print(
        f"Loaded {len(clusters)} clusters "
        f"from {args.clusters_file}"
    )

    extracted_pairs = []
    next_id = args.start_id

    for position, cluster in enumerate(
        clusters,
        start=1,
    ):
        print(
            f"[{position}/{len(clusters)}] "
            f"Cluster {cluster['cluster_id']} "
            f"({cluster['size']} messages)",
            end=" ",
        )

        pairs = extract_from_cluster(
            client,
            cluster,
        )

        for pair in pairs:
            pair["id"] = f"qa{next_id:03d}"
            next_id += 1

        extracted_pairs.extend(pairs)
        print(f"-> {len(pairs)} Q&A pairs")

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            extracted_pairs,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nWrote {len(extracted_pairs)} Q&A pairs "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
