"""Group parsed Telegram messages into conversation clusters."""

import argparse
import json
from collections import defaultdict
from datetime import datetime


def parse_timestamp(timestamp):
    timestamp = timestamp.split(" UTC")[0]
    return datetime.strptime(timestamp, "%d.%m.%Y %H:%M:%S")


def build_clusters(messages, gap_minutes=45):
    messages = [
        dict(message)
        for message in messages
        if message.get("timestamp")
    ]

    for message in messages:
        message["_datetime"] = parse_timestamp(
            message["timestamp"]
        )

    messages.sort(key=lambda message: message["_datetime"])

    parent = {
        message["id"]: message["id"]
        for message in messages
    }

    def find(message_id):
        while parent[message_id] != message_id:
            parent[message_id] = parent[parent[message_id]]
            message_id = parent[message_id]

        return message_id

    def union(first_id, second_id):
        first_root = find(first_id)
        second_root = find(second_id)

        if first_root != second_root:
            parent[first_root] = second_root

    time_groups = defaultdict(list)
    group_number = 0
    previous_message = None

    for message in messages:
        if previous_message:
            gap = (
                message["_datetime"]
                - previous_message["_datetime"]
            ).total_seconds() / 60

            if gap > gap_minutes:
                group_number += 1

        time_groups[group_number].append(message["id"])
        previous_message = message

    for message_ids in time_groups.values():
        for first_id, second_id in zip(
            message_ids,
            message_ids[1:],
        ):
            union(first_id, second_id)

    messages_by_id = {
        message["id"]: message
        for message in messages
    }

    for message in messages:
        reply_to = message.get("reply_to")

        if reply_to and reply_to in messages_by_id:
            union(message["id"], reply_to)

    grouped_messages = defaultdict(list)

    for message in messages:
        cluster_id = find(message["id"])
        grouped_messages[cluster_id].append(message)

    clusters = []

    for cluster_id, members in grouped_messages.items():
        members.sort(
            key=lambda message: message["_datetime"]
        )

        for message in members:
            message.pop("_datetime")

        clusters.append(
            {
                "cluster_id": cluster_id,
                "start": members[0]["timestamp"],
                "end": members[-1]["timestamp"],
                "size": len(members),
                "messages": members,
            }
        )

    clusters.sort(
        key=lambda cluster: parse_timestamp(cluster["start"])
    )
    return clusters


def main():
    parser = argparse.ArgumentParser(
        description="Group Telegram messages into conversations."
    )
    parser.add_argument(
        "input",
        help="JSONL file produced by parse_telegram.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="clusters.jsonl",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=45,
        help="maximum gap between related messages in minutes",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as file:
        messages = [
            json.loads(line)
            for line in file
        ]

    messages = [
        message
        for message in messages
        if message["type"] == "message"
    ]

    clusters = build_clusters(
        messages,
        gap_minutes=args.gap,
    )

    with open(args.output, "w", encoding="utf-8") as file:
        for cluster in clusters:
            file.write(
                json.dumps(cluster, ensure_ascii=False) + "\n"
            )

    print(
        f"{len(clusters)} clusters "
        f"from {len(messages)} messages"
    )

    if clusters:
        sizes = [cluster["size"] for cluster in clusters]
        average_size = sum(sizes) / len(sizes)

        print(
            f"Size range: {min(sizes)}-{max(sizes)}, "
            f"average: {average_size:.1f}"
        )

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
