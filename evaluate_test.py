import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data.dataset import SignRecognitionDataset
from models.sign_recognizer import SignRecognizer


PROJECT_DIR = Path(__file__).resolve().parent
VAL_DIR = PROJECT_DIR / "evaluation_data" / "test"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "sign_recognizer.pt"


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    classes = checkpoint["classes"]

    if checkpoint["input_dim"] != config.FRAME_FEATURE_DIM:
        raise ValueError(
            "Checkpoint input dimension does not match config.py"
        )

    if checkpoint["window_size"] != config.SLIDING_WINDOW_SIZE:
        raise ValueError(
            "Checkpoint window size does not match config.py"
        )

    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )

    dataset = SignRecognitionDataset(
        data_dir=str(VAL_DIR),
        classes=classes,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = SignRecognizer(
        num_classes=len(classes)
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    class_to_idx = {
        name: idx for idx, name in enumerate(classes)
    }

    confusion = np.zeros(
        (len(classes), len(classes)),
        dtype=np.int64,
    )

    records = []
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y, lengths in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x, lengths=lengths)
            probabilities = torch.softmax(logits, dim=1)

            confidences, predictions = probabilities.max(dim=1)

            for i in range(len(y)):
                true_idx = int(y[i].item())
                pred_idx = int(predictions[i].item())
                confidence = float(confidences[i].item())

                path, actual_class = dataset.samples[total]

                confusion[true_idx, pred_idx] += 1
                total += 1
                correct += int(true_idx == pred_idx)

                records.append({
                    "filename": Path(path).name,
                    "actual_class": classes[true_idx],
                    "predicted_class": classes[pred_idx],
                    "correct": true_idx == pred_idx,
                    "confidence": round(confidence, 6),
                })

    accuracy = correct / total if total else 0.0

    print("\n========== VALIDATION RESULTS ==========")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        "Checkpoint recorded validation accuracy: "
        f"{checkpoint['best_val_acc']:.3f}"
    )
    print(f"Samples evaluated: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Recomputed accuracy: {accuracy:.3f}")

    print("\n========== PER-CLASS METRICS ==========")
    print(
        f"{'Class':<15}{'Precision':>12}"
        f"{'Recall':>10}{'F1':>10}{'Support':>10}"
    )

    for idx, name in enumerate(classes):
        tp = confusion[idx, idx]
        predicted_count = confusion[:, idx].sum()
        actual_count = confusion[idx, :].sum()

        precision = (
            tp / predicted_count if predicted_count else 0.0
        )
        recall = (
            tp / actual_count if actual_count else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        print(
            f"{name:<15}{precision:>12.3f}"
            f"{recall:>10.3f}{f1:>10.3f}"
            f"{actual_count:>10}"
        )

    print("\n========== CONFUSION MATRIX ==========")
    print("Rows = actual class; columns = predicted class")
    print("Predicted:", " ".join(f"{name:>10}" for name in classes))

    for idx, name in enumerate(classes):
        values = " ".join(
            f"{value:>10}" for value in confusion[idx]
        )
        print(f"{name:<12}{values}")

    predictions_path = (
        PROJECT_DIR / "test_predictions.csv"
    )

    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "actual_class",
                "predicted_class",
                "correct",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    mistakes_path = (
        PROJECT_DIR / "test_mistakes.csv"
    )

    with mistakes_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "filename",
            "actual_class",
            "predicted_class",
            "correct",
            "confidence",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            row for row in records if not row["correct"]
        )

    confusion_path = (
        PROJECT_DIR / "test_confusion_matrix.csv"
    )

    np.savetxt(
        confusion_path,
        confusion,
        delimiter=",",
        fmt="%d",
        header=(
            "Rows=actual, columns=predicted; class order: "
            + ", ".join(classes)
        ),
        comments="",
    )

    print("\n========== SAVED REPORTS ==========")
    print(f"All predictions: {predictions_path}")
    print(f"Incorrect predictions: {mistakes_path}")
    print(f"Confusion matrix: {confusion_path}")

    print("\n========== LOW-CONFIDENCE MISTAKES ==========")

    mistakes = [
        row for row in records
        if not row["correct"]
    ]

    mistakes.sort(key=lambda row: row["confidence"])

    for row in mistakes[:15]:
        print(
            f"{row['filename']}: "
            f"actual={row['actual_class']}, "
            f"predicted={row['predicted_class']}, "
            f"confidence={row['confidence']:.3f}"
        )

    if not mistakes:
        print("No mistakes found.")


if __name__ == "__main__":
    main()
