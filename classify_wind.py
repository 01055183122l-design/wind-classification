import csv
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


THRESHOLD_MS = 4.0
FEATURE_COLUMNS = (
    "wind_speed_ms",
    "temperature_C",
    "humidity_pct",
    "wind_speed_1h_ago_ms",
    "wind_speed_2h_ago_ms",
)
REQUIRED_COLUMNS = {
    "prediction_time",
    "target_time",
    "actual_wind_speed_1h_ahead_ms",
    *FEATURE_COLUMNS,
}
PREDICTION_COLUMNS = (
    "prediction_time",
    "target_time",
    "actual_wind_speed_1h_ahead_ms",
    "wind_next1h",
    "wind_speed_ms",
    "always_0_prediction",
    "current_wind_threshold_prediction",
    "strong_wind_probability",
    "logistic_regression_prediction",
    "logistic_regression_prediction_020",
)
SCORE_COLUMNS = (
    "method",
    "evaluation_rows",
    "actual_strong_wind",
    "hits",
    "misses",
    "false_alarms",
    "correct_negatives",
    "accuracy",
    "precision",
    "recall",
)


def main():
    dataset_path = Path(__file__).resolve().parent / "wind_dataset.csv"
    predictions_path = dataset_path.with_name("classification_predictions.csv")
    scores_path = dataset_path.with_name("classification_scores.csv")
    training_rows = []
    evaluation_rows = []
    previous_prediction_time = None

    with dataset_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

        for row_number, row in enumerate(reader, start=2):
            prediction_time = datetime.fromisoformat(row["prediction_time"])
            target_time = datetime.fromisoformat(row["target_time"])
            if target_time - prediction_time != timedelta(hours=1):
                raise ValueError(
                    f"Row {row_number}: prediction_time and target_time "
                    "must be exactly one hour apart"
                )
            if (
                previous_prediction_time is not None
                and prediction_time <= previous_prediction_time
            ):
                raise ValueError(
                    f"Row {row_number}: prediction_time is not in increasing order"
                )
            previous_prediction_time = prediction_time

            wind_next1h = float(row["actual_wind_speed_1h_ahead_ms"])
            row["wind_next1h"] = "1" if wind_next1h >= THRESHOLD_MS else "0"

            if datetime(2023, 1, 1) <= target_time < datetime(2024, 12, 1):
                training_rows.append(row)
            elif datetime(2024, 12, 1) <= target_time < datetime(2025, 1, 1):
                evaluation_rows.append(row)

    if not evaluation_rows:
        raise ValueError("No evaluation rows found for target times in December 2024")

    training_features = np.asarray(
        [[float(row[column]) for column in FEATURE_COLUMNS] for row in training_rows],
        dtype=float,
    )
    training_labels = np.asarray(
        [int(row["wind_next1h"]) for row in training_rows], dtype=float
    )
    evaluation_features = np.asarray(
        [[float(row[column]) for column in FEATURE_COLUMNS] for row in evaluation_rows],
        dtype=float,
    )

    feature_means = training_features.mean(axis=0)
    feature_scales = training_features.std(axis=0)
    feature_scales[feature_scales == 0] = 1.0
    training_features = (training_features - feature_means) / feature_scales
    evaluation_features = (evaluation_features - feature_means) / feature_scales

    sample_count = len(training_labels)
    regularization = 1.0 / sample_count
    coefficients = np.zeros(training_features.shape[1], dtype=float)
    intercept = 0.0
    design_matrix = np.column_stack(
        (np.ones(sample_count, dtype=float), training_features)
    )
    step_size = 1.0 / (
        0.25 * np.linalg.norm(design_matrix, ord=2) ** 2 / sample_count
        + regularization
    )
    converged = False
    for _ in range(10_000):
        scores = training_features @ coefficients + intercept
        probabilities = np.exp(-np.logaddexp(0.0, -scores))
        errors = probabilities - training_labels
        coefficient_gradient = (
            training_features.T @ errors / sample_count
            + regularization * coefficients
        )
        intercept_gradient = errors.mean()
        if max(
            np.max(np.abs(coefficient_gradient)), abs(intercept_gradient)
        ) < 1e-8:
            converged = True
            break
        coefficients -= step_size * coefficient_gradient
        intercept -= step_size * intercept_gradient

    if not converged:
        raise RuntimeError("Logistic regression did not converge")

    evaluation_scores = evaluation_features @ coefficients + intercept
    probabilities = np.exp(-np.logaddexp(0.0, -evaluation_scores))
    predictions_050 = (probabilities >= 0.50).astype(int)
    predictions_020 = (probabilities >= 0.20).astype(int)

    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for row, probability, prediction_050, prediction_020 in zip(
            evaluation_rows, probabilities, predictions_050, predictions_020
        ):
            writer.writerow(
                {
                    "prediction_time": row["prediction_time"],
                    "target_time": row["target_time"],
                    "actual_wind_speed_1h_ahead_ms": row[
                        "actual_wind_speed_1h_ahead_ms"
                    ],
                    "wind_next1h": row["wind_next1h"],
                    "wind_speed_ms": row["wind_speed_ms"],
                    "always_0_prediction": 0,
                    "current_wind_threshold_prediction": int(
                        float(row["wind_speed_ms"]) >= THRESHOLD_MS
                    ),
                    "strong_wind_probability": f"{probability:.6f}",
                    "logistic_regression_prediction": int(prediction_050),
                    "logistic_regression_prediction_020": int(prediction_020),
                }
            )

    with predictions_path.open("r", newline="", encoding="utf-8") as csv_file:
        saved_predictions = list(csv.DictReader(csv_file))
    expected_times = [
        (row["prediction_time"], row["target_time"]) for row in evaluation_rows
    ]
    saved_times = [
        (row["prediction_time"], row["target_time"]) for row in saved_predictions
    ]
    if saved_times != expected_times:
        raise ValueError("Saved predictions do not match the December evaluation rows")
    if any(
        int(saved["wind_next1h"]) != int(source["wind_next1h"])
        for saved, source in zip(saved_predictions, evaluation_rows)
    ):
        raise ValueError("Saved prediction labels do not match the evaluation labels")

    actual_labels = [int(row["wind_next1h"]) for row in evaluation_rows]
    actual_strong_wind_count = sum(actual_labels)
    methods = (
        ("always_0", "always_0_prediction"),
        ("current_wind_threshold", "current_wind_threshold_prediction"),
        ("logistic_regression_0.50", "logistic_regression_prediction"),
        ("logistic_regression_0.20", "logistic_regression_prediction_020"),
    )
    score_rows = []
    for method, prediction_column in methods:
        predicted_labels = [int(row[prediction_column]) for row in saved_predictions]
        hits = sum(actual == 1 and predicted == 1 for actual, predicted in zip(
            actual_labels, predicted_labels
        ))
        misses = sum(actual == 1 and predicted == 0 for actual, predicted in zip(
            actual_labels, predicted_labels
        ))
        false_alarms = sum(actual == 0 and predicted == 1 for actual, predicted in zip(
            actual_labels, predicted_labels
        ))
        correct_negatives = sum(
            actual == 0 and predicted == 0
            for actual, predicted in zip(actual_labels, predicted_labels)
        )
        predicted_strong_wind_count = hits + false_alarms
        score_rows.append(
            {
                "method": method,
                "evaluation_rows": len(evaluation_rows),
                "actual_strong_wind": actual_strong_wind_count,
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_negatives": correct_negatives,
                "accuracy": f"{(hits + correct_negatives) / len(evaluation_rows):.6f}",
                "precision": (
                    f"{hits / predicted_strong_wind_count:.6f}"
                    if predicted_strong_wind_count
                    else "해당없음"
                ),
                "recall": (
                    f"{hits / actual_strong_wind_count:.6f}"
                    if actual_strong_wind_count
                    else "해당없음"
                ),
            }
        )

    with scores_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(score_rows)

    for period_name, rows in (
        ("TRAIN (target_time: 2023-01 through 2024-11)", training_rows),
        ("EVALUATION (target_time: 2024-12-01 through 2024-12-31)", evaluation_rows),
    ):
        counts = Counter(row["wind_next1h"] for row in rows)
        print(
            f"{period_name}: total={len(rows)}, "
            f"label_0={counts['0']}, label_1={counts['1']}"
        )

    print(f"Saved predictions to: {predictions_path}")
    print("First 5 prediction rows:")
    with predictions_path.open("r", newline="", encoding="utf-8") as csv_file:
        for row_number, row in enumerate(csv.reader(csv_file)):
            print(",".join(row))
            if row_number == 5:
                break
    print(
        f"Compared the same {len(evaluation_rows)} evaluation rows; "
        f"actual strong-wind rows={actual_strong_wind_count}."
    )
    print(f"Saved scores to: {scores_path}")
    print(
        "method,evaluation_rows,actual_strong_wind,hits,misses,false_alarms,"
        "correct_negatives,accuracy,precision,recall"
    )
    for row in score_rows:
        print(",".join(str(row[column]) for column in SCORE_COLUMNS))


if __name__ == "__main__":
    main()
