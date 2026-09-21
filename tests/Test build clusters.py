"""Tests for Telegram conversation clustering."""

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_clusters import (  # noqa: E402
    build_clusters,
    parse_timestamp,
)


def make_message(
    message_id,
    timestamp,
    reply_to=None,
):
    return {
        "id": message_id,
        "timestamp": timestamp,
        "reply_to": reply_to,
        "text": f"text {message_id}",
    }


def message_ids(cluster):
    return [
        message["id"]
        for message in cluster["messages"]
    ]


def test_ignores_utc_suffix():
    with_suffix = parse_timestamp(
        "12.03.2024 18:30:15 UTC+01:00"
    )
    without_suffix = parse_timestamp(
        "12.03.2024 18:30:15"
    )

    assert with_suffix == without_suffix


def test_groups_messages_within_time_limit():
    messages = [
        make_message("1", "01.03.2024 10:00:00"),
        make_message("2", "01.03.2024 10:20:00"),
        make_message("3", "01.03.2024 10:50:00"),
    ]

    clusters = build_clusters(
        messages,
        gap_minutes=45,
    )

    assert len(clusters) == 1
    assert message_ids(clusters[0]) == ["1", "2", "3"]


def test_splits_messages_after_large_gap():
    messages = [
        make_message("1", "01.03.2024 10:00:00"),
        make_message("2", "01.03.2024 10:10:00"),
        make_message("3", "01.03.2024 14:00:00"),
    ]

    clusters = build_clusters(
        messages,
        gap_minutes=45,
    )

    assert [
        message_ids(cluster)
        for cluster in clusters
    ] == [["1", "2"], ["3"]]


def test_reply_connects_messages_across_time_gap():
    messages = [
        make_message("1", "01.03.2024 10:00:00"),
        make_message(
            "2",
            "02.03.2024 09:00:00",
            reply_to="1",
        ),
    ]

    clusters = build_clusters(
        messages,
        gap_minutes=45,
    )

    assert len(clusters) == 1
    assert message_ids(clusters[0]) == ["1", "2"]


def test_ignores_reply_to_unknown_message():
    messages = [
        make_message(
            "1",
            "01.03.2024 10:00:00",
            reply_to="999",
        )
    ]

    clusters = build_clusters(
        messages,
        gap_minutes=45,
    )

    assert len(clusters) == 1


def test_sorts_clusters_chronologically():
    messages = [
        make_message(
            "later",
            "02.01.2022 10:00:00",
        ),
        make_message(
            "earlier",
            "15.06.2021 10:00:00",
        ),
    ]

    clusters = build_clusters(
        messages,
        gap_minutes=45,
    )

    assert [
        message_ids(cluster)
        for cluster in clusters
    ] == [["earlier"], ["later"]]


def test_drops_messages_without_timestamp():
    messages = [
        make_message("1", "01.03.2024 10:00:00"),
        make_message("2", None),
    ]

    clusters = build_clusters(
        messages,
        gap_minutes=45,
    )

    assert sum(
        cluster["size"]
        for cluster in clusters
    ) == 1
