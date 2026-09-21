"""Assign topic categories to Q&A entries."""

import argparse
import json
import re


CATEGORY_RULES = [
    (
        "Pass & Botschaft",
        (
            r"pass|reisepass|botschaft|konsulat|mikhak|miykhak|"
            r"wehrpflicht|ausreise|mehr-e khorooj|shenasnameh"
        ),
    ),
    (
        "Aufenthaltstitel & Visum",
        (
            r"aufenthalt|visum|fiktionsbescheinigung|blaue karte|"
            r"niederlassung|staatsangehörig|einbürgerung|18d|18b"
        ),
    ),
    (
        "Bank & Finanzen",
        r"bank|konto|kredit|geld|überweisung|steuer|zoll",
    ),
    (
        "Krankenversicherung",
        r"krankenversicherung|mawista|tk\b|aok|care concept|dr\.? walter",
    ),
    (
        "Wohnen & Meldung",
        (
            r"wohnung|wg-|wg |meldung|anmeldung|zwischenmiete|"
            r"wohnheim|vermieter|wohnungsgeber|umzug"
        ),
    ),
    (
        "Uni & Studium",
        (
            r"studien|semester|uni|prüfung|klausur|immatrikul|"
            r"studip|tune|bibliothek|urlaubssemester|"
            r"integrationskurs|sprachkurs"
        ),
    ),
    (
        "Job & Ausbildung",
        (
            r"job|arbeit|gehalt|zenjob|ausbildung|bewerbung|"
            r"lebenslauf|freelance|selbständig"
        ),
    ),
    (
        "Führerschein & Verkehr",
        (
            r"führerschein|hvv|fahrrad|semesterticket|"
            r"deutschlandticket|bahn|flug|gepäck"
        ),
    ),
    (
        "Gesundheit & Alltag",
        r"arzt|gesundheit|zahn|apotheke|medikament",
    ),
    (
        "Konnektivität / Krise",
        r"konnektivität|vpn|internet|krise",
    ),
]


def categorize(topic: str) -> str:
    topic = (topic or "").lower()

    for category, pattern in CATEGORY_RULES:
        if re.search(pattern, topic):
            return category

    return "Sonstiges"


def build_categorized(entries: list) -> dict:
    entries_by_category = {}

    for entry in entries:
        category = categorize(entry.get("topic", ""))
        entry["category"] = category

        entries_by_category.setdefault(
            category,
            [],
        ).append(entry)

    return entries_by_category


def main():
    parser = argparse.ArgumentParser(
        description="Group Q&A entries by topic category."
    )
    parser.add_argument(
        "qa_file",
        help="JSON file containing the Q&A entries",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="qa_categorized.json",
    )
    args = parser.parse_args()

    with open(args.qa_file, encoding="utf-8") as file:
        entries = json.load(file)

    entries_by_category = build_categorized(entries)

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            entries_by_category,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Category breakdown:")

    sorted_categories = sorted(
        entries_by_category.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for category, items in sorted_categories:
        answered_count = sum(
            item.get("answer_type") != "unanswered"
            for item in items
        )
        print(
            f"  {category}: {len(items)} total "
            f"({answered_count} answered)"
        )


if __name__ == "__main__":
    main()
