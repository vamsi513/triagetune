#!/usr/bin/env python3
"""Prepare BANKING77 splits and audit cross-split text leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.90


@dataclass(frozen=True)
class Record:
    text: str
    category: str
    source_split: str
    source_index: int


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=project_root / "data" / "raw" / "train.csv",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=project_root / "data" / "raw" / "test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=project_root / "reports" / "data_preparation_report.json",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    )
    return parser.parse_args()


def canonical_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def comparison_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", canonical_text(text)).strip()


def canonical_category(category: str) -> str:
    value = unicodedata.normalize("NFKC", category).casefold()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def load_csv(path: Path, source_split: str) -> list[Record]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "category"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        return [
            Record(
                text=row["text"],
                category=canonical_category(row["category"]),
                source_split=source_split,
                source_index=index,
            )
            for index, row in enumerate(reader)
        ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return str(resolved)


def duplicate_audit(records: Iterable[Record]) -> dict[str, object]:
    records = list(records)
    by_raw: dict[str, list[Record]] = defaultdict(list)
    by_canonical: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_raw[record.text].append(record)
        by_canonical[canonical_text(record.text)].append(record)

    raw_groups = [group for group in by_raw.values() if len(group) > 1]
    canonical_groups = [group for group in by_canonical.values() if len(group) > 1]
    raw_duplicate_keys = {record.text for group in raw_groups for record in group}
    canonical_only_groups = [
        group
        for group in canonical_groups
        if not all(record.text in raw_duplicate_keys for record in group)
    ]
    conflicting_groups = [
        group
        for group in canonical_groups
        if len({record.category for record in group}) > 1
    ]

    def crosses_source_split(group: list[Record]) -> bool:
        return len({record.source_split for record in group}) > 1

    return {
        "raw_duplicate_groups": len(raw_groups),
        "raw_duplicate_excess_records": sum(len(group) - 1 for group in raw_groups),
        "raw_cross_source_split_groups": sum(map(crosses_source_split, raw_groups)),
        "canonical_only_duplicate_groups": len(canonical_only_groups),
        "canonical_only_duplicate_excess_records": sum(
            len(group) - 1 for group in canonical_only_groups
        ),
        "canonical_cross_source_split_groups": sum(
            map(crosses_source_split, canonical_groups)
        ),
        "conflicting_label_duplicate_groups": len(conflicting_groups),
    }


def deduplicate_preserving_test(
    train_records: list[Record], test_records: list[Record]
) -> tuple[list[Record], list[Record], dict[str, int]]:
    seen = {canonical_text(record.text) for record in test_records}
    kept_test = list(test_records)
    kept_train: list[Record] = []
    removed = Counter()

    for record in train_records:
        key = canonical_text(record.text)
        if key in seen:
            removed["train"] += 1
        else:
            seen.add(key)
            kept_train.append(record)

    return kept_train, kept_test, {"train": removed["train"], "test": removed["test"]}


def stratified_split(
    records: list[Record], validation_fraction: float, seed: int
) -> tuple[list[Record], list[Record]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must be between 0 and 1")

    rng = random.Random(seed)
    by_category: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_category[record.category].append(record)
    for category_records in by_category.values():
        rng.shuffle(category_records)

    target_validation_size = round(len(records) * validation_fraction)
    allocations = {
        category: math.floor(len(category_records) * validation_fraction)
        for category, category_records in by_category.items()
    }
    remaining = target_validation_size - sum(allocations.values())
    tie_breakers = {category: rng.random() for category in by_category}
    ranked_categories = sorted(
        by_category,
        key=lambda category: (
            -((len(by_category[category]) * validation_fraction) % 1),
            tie_breakers[category],
            category,
        ),
    )
    for category in ranked_categories[:remaining]:
        allocations[category] += 1

    train: list[Record] = []
    validation: list[Record] = []
    for category, category_records in sorted(by_category.items()):
        split_at = allocations[category]
        validation.extend(category_records[:split_at])
        train.extend(category_records[split_at:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def character_ngrams(value: str, size: int = 3) -> set[str]:
    compact = f" {comparison_text(value)} "
    if len(compact) <= size:
        return {compact}
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def find_near_duplicates(
    left_name: str,
    left: list[Record],
    right_name: str,
    right: list[Record],
    threshold: float,
) -> list[dict[str, object]]:
    right_values = [comparison_text(record.text) for record in right]
    right_grams = [character_ngrams(value) for value in right_values]
    postings: dict[str, list[int]] = defaultdict(list)
    for index, grams in enumerate(right_grams):
        for gram in grams:
            postings[gram].append(index)

    findings: list[dict[str, object]] = []
    for left_index, left_record in enumerate(left):
        left_value = comparison_text(left_record.text)
        grams = character_ngrams(left_value)
        available = [gram for gram in grams if gram in postings]
        rarest = sorted(available, key=lambda gram: (len(postings[gram]), gram))[:24]
        candidates = {index for gram in rarest for index in postings[gram]}
        if len(left_value) < 8:
            candidates.update(range(len(right)))

        for right_index in candidates:
            right_value = right_values[right_index]
            maximum_possible = (
                2 * min(len(left_value), len(right_value))
                / max(1, len(left_value) + len(right_value))
            )
            if maximum_possible < threshold:
                continue
            if left_value == right_value:
                continue
            intersection_size = len(grams & right_grams[right_index])
            similarity = 2 * intersection_size / max(
                1, len(grams) + len(right_grams[right_index])
            )
            if similarity >= threshold:
                right_record = right[right_index]
                findings.append(
                    {
                        "left_split": left_name,
                        "left_index": left_index,
                        "left_category": left_record.category,
                        "left_text": left_record.text,
                        "right_split": right_name,
                        "right_index": right_index,
                        "right_category": right_record.category,
                        "right_text": right_record.text,
                        "similarity": round(similarity, 6),
                    }
                )
    return sorted(
        findings,
        key=lambda item: (
            -float(item["similarity"]),
            str(item["left_split"]),
            int(item["left_index"]),
            int(item["right_index"]),
        ),
    )


def write_split(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["text", "category"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {"text": record.text, "category": record.category} for record in records
        )


def write_near_duplicate_report(path: Path, findings: list[dict[str, object]]) -> None:
    fields = [
        "left_split",
        "left_index",
        "left_category",
        "left_text",
        "right_split",
        "right_index",
        "right_category",
        "right_text",
        "similarity",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(findings)


def class_summary(records: list[Record]) -> dict[str, object]:
    counts = Counter(record.category for record in records)
    return {
        "records": len(records),
        "classes": len(counts),
        "minimum_class_count": min(counts.values()),
        "maximum_class_count": max(counts.values()),
        "category_counts": dict(sorted(counts.items())),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    train_source = load_csv(args.train_file, "official_train")
    test_source = load_csv(args.test_file, "official_test")
    source_audit = duplicate_audit([*train_source, *test_source])
    deduplicated_train, deduplicated_test, removed = deduplicate_preserving_test(
        train_source, test_source
    )
    train, validation = stratified_split(
        deduplicated_train, args.validation_fraction, args.seed
    )

    split_pairs = [
        ("train", train, "validation", validation),
        ("train", train, "test", deduplicated_test),
        ("validation", validation, "test", deduplicated_test),
    ]
    near_duplicates: list[dict[str, object]] = []
    near_duplicate_counts: dict[str, int] = {}
    for left_name, left, right_name, right in split_pairs:
        findings = find_near_duplicates(
            left_name, left, right_name, right, args.near_duplicate_threshold
        )
        near_duplicates.extend(findings)
        near_duplicate_counts[f"{left_name}--{right_name}"] = len(findings)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_split(args.output_dir / "train.csv", train)
    write_split(args.output_dir / "validation.csv", validation)
    write_split(args.output_dir / "test.csv", deduplicated_test)
    write_near_duplicate_report(
        args.output_dir / "near_duplicate_pairs.csv", near_duplicates
    )

    report = {
        "dataset": "BANKING77",
        "source_files": {
            "train": {
                "path": portable_path(args.train_file, project_root),
                "sha256": sha256_file(args.train_file),
                "records": len(train_source),
            },
            "test": {
                "path": portable_path(args.test_file, project_root),
                "sha256": sha256_file(args.test_file),
                "records": len(test_source),
            },
        },
        "parameters": {
            "seed": args.seed,
            "validation_fraction_of_official_train": args.validation_fraction,
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "near_duplicate_metric": "Sørensen-Dice similarity over character trigrams after lowercase alphanumeric normalization",
            "near_duplicate_candidate_method": "24 rarest shared character trigrams",
        },
        "exact_duplicate_audit": source_audit,
        "removed_duplicate_records": removed,
        "test_preservation": {
            "official_test_records_retained": len(deduplicated_test),
            "canonical_duplicate_excess_records_retained_within_test": (
                len(test_source)
                - len({canonical_text(record.text) for record in test_source})
            ),
        },
        "splits": {
            "train": class_summary(train),
            "validation": class_summary(validation),
            "test": class_summary(deduplicated_test),
        },
        "near_duplicate_pairs": {
            "total": len(near_duplicates),
            "by_split_pair": near_duplicate_counts,
            "unique_records_flagged_by_split": {
                split_name: len(
                    {
                        (item["left_split"], item["left_index"])
                        for item in near_duplicates
                        if item["left_split"] == split_name
                    }
                    | {
                        (item["right_split"], item["right_index"])
                        for item in near_duplicates
                        if item["right_split"] == split_name
                    }
                )
                for split_name in ("train", "validation", "test")
            },
            "same_category": sum(
                item["left_category"] == item["right_category"]
                for item in near_duplicates
            ),
            "different_category": sum(
                item["left_category"] != item["right_category"]
                for item in near_duplicates
            ),
            "report": portable_path(
                args.output_dir / "near_duplicate_pairs.csv", project_root
            ),
        },
    }
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
