from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data.dataset import SignRecognitionDataset
from models.sign_recognizer import SignRecognizer


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_DIR / "dataset"
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
        raise ValueError("Checkpoint input dimension mismatch.")

    if checkpoint["window_size"] != config.SLIDING_WINDOW_SIZE:
        raise ValueError("Checkpoint window size mismatch.")

    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )

    dataset = SignRecognitionDataset(
        data_dir=str(TRAIN_DIR),
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

    confusion = np.zeros(
        (len(classes), len(classes)),
        dtype=np.int64,
    )

    total = 0
    correct = 0

    with torch.no_grad():
        for x, y, lengths in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x, lengths=lengths)
            predictions = logits.argmax(dim=1)

            for true_idx, pred_idx in zip(
                y.cpu().tolist(),
                predictions.cpu().tolist(),
            ):
                confusion[true_idx, pred_idx] += 1
                total += 1
                correct += int(true_idx == pred_idx)

    accuracy = correct / total if total else 0.0

    print("\n========== TRAINING DATASET EVALUATION ==========")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Samples evaluated: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Training accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")

    print("\n========== PER-CLASS ACCURACY ==========")

    for idx, name in enumerate(classes):
        count = confusion[idx].sum()
        class_correct = confusion[idx, idx]
        class_accuracy = (
            class_correct / count if count else 0.0
        )

        print(
            f"{name:<15} "
            f"{class_correct:>3}/{count:<3} "
            f"({class_accuracy * 100:.1f}%)"
        )

    print("\n========== CONFUSION MATRIX ==========")
    print("Rows = actual class; columns = predicted class")
    print("Classes:", classes)
    print(confusion)


if __name__ == "__main__":
    main()
