"""
train_recognizer.py
---------------------
Training loop for the CNN+BiLSTM sign recognizer. Reads landmark samples
collected via data/collect_data.py, splits into train/val, trains, and
saves the best checkpoint + class label mapping.

Usage:
    python -m train.train_recognizer --epochs 50
"""

import argparse
import json
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import config
from data.dataset import SignRecognitionDataset
from models.sign_recognizer import SignRecognizer


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def train(epochs, batch_size, lr, val_split=0.15):
    set_seed()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = SignRecognitionDataset()
    if len(dataset) == 0:
        raise RuntimeError(
            f"No training samples found in {config.DATA_DIR}. "
            "Run data/collect_data.py first to record some signs."
        )
    num_classes = len(dataset.classes)
    print(f"Found {len(dataset)} samples across {num_classes} classes: {dataset.classes}")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SignRecognizer(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(config.RECOGNIZER_CKPT), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y, lengths in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths=lengths)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += x.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        print(f"Epoch {epoch:3d}/{epochs} | loss {total_loss/total:.4f} "
              f"| train_acc {train_acc:.3f} | val_acc {val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": dataset.classes,
                "num_classes": num_classes,
            }, config.RECOGNIZER_CKPT)

            with open(config.GLOSS_VOCAB_PATH, "w") as f:
                json.dump(dataset.classes, f, indent=2)

    print(f"Best val accuracy: {best_val_acc:.3f}. Checkpoint saved to {config.RECOGNIZER_CKPT}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y, lengths in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, lengths=lengths)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_RECOGNIZER)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    train(args.epochs, args.batch_size, args.lr)
