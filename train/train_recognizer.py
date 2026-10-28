"""
Training script for the CNN+BiLSTM sign recognizer.

Training:   dataset/
Validation: evaluation_data/validation/
Testing:    evaluation_data/test/ (reserved for final evaluation)

Run from the project root:
    python -m train.train_recognizer --epochs 50
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data.dataset import SignRecognitionDataset
from models.sign_recognizer import SignRecognizer


PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "dataset"
VAL_DIR = PROJECT_DIR / "evaluation_data" / "validation"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(epochs, batch_size, lr):
    set_seed()

    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    print(f"Training device: {device}")

    # Load training data and establish the class-index mapping.
    train_ds = SignRecognitionDataset(data_dir=str(TRAIN_DIR))

    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found in {TRAIN_DIR}")

    # Reuse the training class ordering for validation.
    val_ds = SignRecognitionDataset(
        data_dir=str(VAL_DIR),
        classes=train_ds.classes,
    )

    if len(val_ds) == 0:
        raise RuntimeError(f"No validation samples found in {VAL_DIR}")

    if train_ds.classes != val_ds.classes:
        raise RuntimeError("Training and validation class mappings differ.")

    num_classes = len(train_ds.classes)

    print(f"Classes: {train_ds.classes}")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SignRecognizer(num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    checkpoint_path = Path(config.RECOGNIZER_CKPT)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_path = Path(config.GLOSS_VOCAB_PATH)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y, lengths in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x, lengths=lengths)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += x.size(0)

        train_loss = total_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss {train_loss:.4f} | "
            f"train_acc {train_acc:.3f} | "
            f"val_loss {val_loss:.4f} | "
            f"val_acc {val_acc:.3f} | "
            f"lr {current_lr:.2e}"
        )

        # Save only when validation accuracy improves.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": train_ds.classes,
                    "num_classes": num_classes,
                    "input_dim": config.FRAME_FEATURE_DIM,
                    "window_size": config.SLIDING_WINDOW_SIZE,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(train_ds.classes, f, indent=2)

            print(f"  Saved improved checkpoint: {checkpoint_path}")

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Checkpoint: {checkpoint_path}")


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y, lengths in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x, lengths=lengths)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    if total == 0:
        return 0.0, 0.0

    return total_loss / total, correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS_RECOGNIZER,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.LEARNING_RATE,
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")

    if args.lr <= 0:
        parser.error("--lr must be positive")

    train(args.epochs, args.batch_size, args.lr)

