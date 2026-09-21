"""Tests for the Q&A categorization rules."""

import re
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from categorize_qa import (  # noqa: E402
    CATEGORY_RULES,
    build_categorized,
    categorize,
)


@pytest.mark.parametrize(
    ("topic", "expected"),
    [
        (
            "Bank / Konto-Eröffnung ohne Meldung",
            "Bank & Finanzen",
        ),
        (
            "Fiktionsbescheinigung: wo und wie beantragen",
            "Aufenthaltstitel & Visum",
        ),
        (
            "Irgendwas ganz Unbekanntes und Zufälliges",
            "Sonstiges",
        ),
        (
            "",
            "Sonstiges",
        ),
    ],
)
def test_categorizes_topics(topic, expected):
    assert categorize(topic) == expected


def test_category_matching_is_case_insensitive():
    assert categorize("BANK KONTO") == categorize(
        "bank konto"
    )


def test_groups_entries_by_category():
    entries = [
        {
            "id": "1",
            "topic": "Bank Konto",
            "answer_type": "explicit_reply",
            "answer": "x",
        },
        {
            "id": "2",
            "topic": "Fiktionsbescheinigung",
            "answer_type": "explicit_reply",
            "answer": "y",
        },
        {
            "id": "3",
            "topic": "Zufälliges Thema",
            "answer_type": "unanswered",
            "answer": None,
        },
    ]

    categorized = build_categorized(entries)

    assert categorized["Bank & Finanzen"][0]["id"] == "1"
    assert (
        categorized["Aufenthaltstitel & Visum"][0]["id"]
        == "2"
    )
    assert categorized["Sonstiges"][0]["id"] == "3"
    assert entries[0]["category"] == "Bank & Finanzen"


def test_category_patterns_are_valid():
    for _, pattern in CATEGORY_RULES:
        re.compile(pattern)
