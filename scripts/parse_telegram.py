"""Convert Telegram Desktop HTML exports to JSONL."""

import argparse
import json
import re
import sys

from bs4 import BeautifulSoup


def parse_file(path, sender_state):
    with open(path, encoding="utf-8") as file:
        soup = BeautifulSoup(file.read(), "lxml")

    records = []
    current_date = None

    for message_div in soup.select("div.message"):
        classes = message_div.get("class", [])
        message_id = (
            message_div.get("id", "")
            .replace("message", "")
            .strip()
            or None
        )

        if "service" in classes:
            body = message_div.select_one(".body.details")
            text = body.get_text(strip=True) if body else ""

            if re.match(r"^\d{1,2} \w+ \d{4}$", text):
                current_date = text

            records.append(
                {
                    "id": message_id,
                    "type": "service",
                    "date": current_date,
                    "text": text,
                }
            )
            continue

        body = message_div.select_one("div.body")
        if body is None:
            continue

        sender_element = body.select_one("div.from_name")
        if sender_element:
            sender = sender_element.get_text(strip=True)
            sender_state["sender"] = sender
        else:
            sender = sender_state.get("sender")

        date_element = body.select_one("div.date.details")
        timestamp = (
            date_element.get("title")
            if date_element
            else None
        )

        reply_to = None
        reply_element = body.select_one("div.reply_to")

        if reply_element:
            link = reply_element.select_one("a")
            href = link.get("href", "") if link else ""

            if href.startswith("#go_to_message"):
                reply_to = href.replace("#go_to_message", "")

        forwarded_element = body.select_one("div.forwarded.body")
        forwarded = forwarded_element is not None
        forwarded_from = None

        if forwarded_element:
            name_element = forwarded_element.select_one("div.from_name")

            if name_element:
                # Use only the direct text to avoid including the date.
                name = name_element.find(string=True, recursive=False)
                forwarded_from = (
                    name.strip()
                    if name
                    else name_element.get_text(strip=True)
                )

        media_type = None
        media_element = body.select_one("div.media")

        if media_element:
            for class_name in media_element.get("class", []):
                if class_name.startswith("media_"):
                    media_type = class_name.replace("media_", "")

        if body.select_one("div.media_poll"):
            media_type = "poll"

        text_element = body.select_one("div.text")
        text = (
            text_element.get_text("\n", strip=True)
            if text_element
            else ""
        )

        if not text and not media_type and not forwarded:
            continue

        records.append(
            {
                "id": message_id,
                "type": "message",
                "date": current_date,
                "timestamp": timestamp,
                "sender": sender,
                "reply_to": reply_to,
                "forwarded": forwarded,
                "forwarded_from": forwarded_from,
                "media_type": media_type,
                "text": text,
            }
        )

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Convert Telegram HTML exports to JSONL."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="messagesN.html files in chronological order",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="messages.jsonl",
    )
    args = parser.parse_args()

    sender_state = {}
    all_records = []

    for path in args.files:
        records = parse_file(path, sender_state)
        all_records.extend(records)
        print(
            f"{path}: {len(records)} records",
            file=sys.stderr,
        )

    with open(args.output, "w", encoding="utf-8") as file:
        for record in all_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    message_count = sum(
        record["type"] == "message"
        for record in all_records
    )
    service_count = sum(
        record["type"] == "service"
        for record in all_records
    )

    print(
        f"Total: {len(all_records)} records "
        f"({message_count} messages, "
        f"{service_count} service messages)",
        file=sys.stderr,
    )
    print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
