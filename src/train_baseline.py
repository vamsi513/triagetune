#!/usr/bin/env python3
"""Train and evaluate reproducible non-neural baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=("majority", "tfidf_logreg"),
        default="majority",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=project_root / "data" / "processed" / "train.csv",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=project_root / "data" / "processed" / "validation.csv",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Defaults to a baseline-specific file under reports/.",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=project_root / "artifacts" / "tfidf_logreg.joblib",
    )
    parser.add_argument(
        "--confusion-matrix-file",
        type=Path,
        default=project_root / "reports" / "tfidf_logreg_confusion_matrix.csv",
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=project_root / "reports" / "tfidf_logreg_validation_predictions.csv",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "category"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        records = [
            {"text": row["text"], "category": row["category"]} for row in reader
        ]
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


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


def classification_metrics(
    expected: list[str], predicted: list[str], labels: list[str]
) -> dict[str, object]:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted labels must have equal lengths")

    correct = sum(actual == prediction for actual, prediction in zip(expected, predicted))
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        true_positive = sum(
            actual == label and prediction == label
            for actual, prediction in zip(expected, predicted)
        )
        false_positive = sum(
            actual != label and prediction == label
            for actual, prediction in zip(expected, predicted)
        )
        false_negative = sum(
            actual == label and prediction != label
            for actual, prediction in zip(expected, predicted)
        )
        support = sum(actual == label for actual in expected)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "accuracy": correct / len(expected),
        "macro_precision": sum(
            float(values["precision"]) for values in per_class.values()
        )
        / len(labels),
        "macro_recall": sum(float(values["recall"]) for values in per_class.values())
        / len(labels),
        "macro_f1": sum(float(values["f1"]) for values in per_class.values())
        / len(labels),
        "correct": correct,
        "total": len(expected),
        "per_class": per_class,
    }


def run_majority(train_labels: list[str], validation_labels: list[str]) -> dict[str, object]:
    training_counts = Counter(train_labels)
    largest_count = max(training_counts.values())
    tied_classes = sorted(
        label for label, count in training_counts.items() if count == largest_count
    )
    majority_class = tied_classes[0]
    predictions = [majority_class] * len(validation_labels)
    labels = sorted(set(train_labels) | set(validation_labels))
    return {
        "baseline": "majority_class",
        "target": "category",
        "selection_source": "training_split_only",
        "majority_class": majority_class,
        "majority_class_training_count": largest_count,
        "majority_class_validation_count": Counter(validation_labels)[majority_class],
        "tied_training_classes": tied_classes,
        "training_records": len(train_labels),
        "validation_records": len(validation_labels),
        "number_of_classes": len(labels),
        "metrics": classification_metrics(validation_labels, predictions, labels),
    }


def write_confusion_matrix(
    path: Path, labels: list[str], matrix: list[list[int]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def write_predictions(
    path: Path,
    records: list[dict[str, str]],
    predictions: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "actual", "predicted", "correct"],
            lineterminator="\n",
        )
        writer.writeheader()
        for record, prediction in zip(records, predictions):
            writer.writerow(
                {
                    "text": record["text"],
                    "actual": record["category"],
                    "predicted": prediction,
                    "correct": record["category"] == prediction,
                }
            )


def run_tfidf_logreg(
    train_records: list[dict[str, str]],
    validation_records: list[dict[str, str]],
    model_file: Path,
    confusion_matrix_file: Path,
    predictions_file: Path,
) -> dict[str, object]:
    import joblib
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.pipeline import Pipeline

    train_texts = [record["text"] for record in train_records]
    train_labels = [record["category"] for record in train_records]
    validation_texts = [record["text"] for record in validation_records]
    validation_labels = [record["category"] for record in validation_records]
    labels = sorted(set(train_labels) | set(validation_labels))

    configuration = {
        "vectorizer": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "min_df": 2,
            "sublinear_tf": True,
            "norm": "l2",
        },
        "classifier": {
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": None,
            "random_state": 42,
        },
    }
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )

    started = time.perf_counter()
    pipeline.fit(train_texts, train_labels)
    training_seconds = time.perf_counter() - started
    predictions = pipeline.predict(validation_texts).tolist()
    metrics = classification_metrics(validation_labels, predictions, labels)
    matrix_array = confusion_matrix(validation_labels, predictions, labels=labels)
    matrix = matrix_array.tolist()

    confusions = []
    for actual_index, actual in enumerate(labels):
        for predicted_index, predicted in enumerate(labels):
            count = matrix[actual_index][predicted_index]
            if actual != predicted and count:
                confusions.append(
                    {"actual": actual, "predicted": predicted, "count": count}
                )
    confusions.sort(key=lambda item: (-int(item["count"]), item["actual"], item["predicted"]))

    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_file)
    write_confusion_matrix(confusion_matrix_file, labels, matrix)
    write_predictions(predictions_file, validation_records, predictions)
    classifier = pipeline.named_steps["classifier"]
    vectorizer = pipeline.named_steps["tfidf"]

    return {
        "baseline": "tfidf_logistic_regression",
        "target": "category",
        "training_records": len(train_records),
        "validation_records": len(validation_records),
        "number_of_classes": len(labels),
        "configuration": configuration,
        "runtime": {
            "scikit_learn_version": sklearn.__version__,
            "training_seconds": training_seconds,
            "features": len(vectorizer.get_feature_names_out()),
            "iterations_by_class": classifier.n_iter_.tolist(),
            "converged_before_max_iterations": bool(
                max(classifier.n_iter_) < configuration["classifier"]["max_iter"]
            ),
        },
        "metrics": metrics,
        "top_confusions": confusions[:20],
        "artifacts": {
            "model_file": str(model_file),
            "confusion_matrix_file": str(confusion_matrix_file),
            "predictions_file": str(predictions_file),
        },
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    train_records = load_records(args.train_file)
    validation_records = load_records(args.validation_file)
    if args.baseline == "majority":
        result = run_majority(
            [record["category"] for record in train_records],
            [record["category"] for record in validation_records],
        )
        default_report = Path(__file__).resolve().parents[1] / "reports" / "majority_baseline_validation.json"
    else:
        result = run_tfidf_logreg(
            train_records,
            validation_records,
            args.model_file,
            args.confusion_matrix_file,
            args.predictions_file,
        )
        default_report = project_root / "reports" / "tfidf_logreg_validation.json"
        result["artifacts"] = {
            name: portable_path(Path(path), project_root)
            for name, path in result["artifacts"].items()
        }
    result["inputs"] = {
        "train_file": portable_path(args.train_file, project_root),
        "train_sha256": sha256_file(args.train_file),
        "validation_file": portable_path(args.validation_file, project_root),
        "validation_sha256": sha256_file(args.validation_file),
    }
    report_file = args.report_file or default_report
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
