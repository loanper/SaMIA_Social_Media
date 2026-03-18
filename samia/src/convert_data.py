import argparse
import csv
import json
import os

# Increase CSV field size limit to handle large fields on Windows.
csv.field_size_limit(2147483647)


def convert_csv_to_jsonl(
    csv_path,
    output_path,
    text_column_name,
    min_length=32,
    max_rows=None,
    source_name=None,
):
    print(f"Converting {csv_path} to {output_path}...")
    data = []

    with open(csv_path, "r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = (row.get(text_column_name) or "").strip()
            if not text:
                continue

            words = text.split()
            if len(words) < min_length:
                continue

            truncated_text = " ".join(words[:min_length])

            # Membership is unknown for Facebook/TikTok custom data.
            # Do not emit synthetic supervision labels such as label=0.
            entry = {
                "input": truncated_text,
                "membership": "unknown",
                "source": source_name or os.path.basename(csv_path),
                "word_count": min_length,
            }
            data.append(entry)

            if max_rows is not None and len(data) >= max_rows:
                break

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for entry in data:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    print(f"Wrote {len(data)} records.")


def build_default_paths(project_root):
    return {
        "facebook": {
            "csv": os.path.join(project_root, "..", "EDA", "Facebook-datasets.csv"),
            "jsonl": os.path.join(project_root, "custom_data", "facebook_32.jsonl"),
            "source": "facebook",
        },
        "tiktok": {
            "csv": os.path.join(project_root, "..", "EDA", "TikTok-datasets.csv"),
            "jsonl": os.path.join(project_root, "custom_data", "tiktok_32.jsonl"),
            "source": "tiktok",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CSV social datasets to membership-unknown JSONL files.",
    )
    parser.add_argument(
        "--dataset",
        choices=["facebook", "tiktok", "all"],
        default="all",
        help="Dataset to convert.",
    )
    parser.add_argument(
        "--text_column",
        default="comment_text",
        help="CSV column containing text.",
    )
    parser.add_argument(
        "--text_length",
        default=32,
        type=int,
        help="Truncated text length in words.",
    )
    parser.add_argument(
        "--max_rows",
        default=None,
        type=int,
        help="Optional cap for quick experiments.",
    )
    parser.add_argument(
        "--small_suffix",
        default="",
        type=str,
        help="Optional suffix for output filename, e.g. _small.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    defaults = build_default_paths(base_dir)

    targets = [args.dataset] if args.dataset != "all" else ["facebook", "tiktok"]
    for name in targets:
        csv_path = defaults[name]["csv"]
        output_path = defaults[name]["jsonl"].replace(
            "_32.jsonl", f"_{args.text_length}{args.small_suffix}.jsonl"
        )
        convert_csv_to_jsonl(
            csv_path=csv_path,
            output_path=output_path,
            text_column_name=args.text_column,
            min_length=args.text_length,
            max_rows=args.max_rows,
            source_name=defaults[name]["source"],
        )
