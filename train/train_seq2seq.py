"""
train_seq2seq.py
------------------
Trains the Gloss2SentenceTransformer on a TSV file of
(gloss_sequence \t target_sentence) pairs.

Example TSV rows (data/gloss_sentence_pairs.tsv):
    i school go yesterday        I went to school yesterday.
    you name what                What is your name?
    i hungry very                I am very hungry.

Usage:
    python -m train.train_seq2seq --data data/gloss_sentence_pairs.tsv --epochs 100
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import config
from data.dataset import Gloss2SentenceDataset, build_vocabs_from_tsv
from models.seq2seq_transformer import Gloss2SentenceTransformer


def train(tsv_path, epochs, batch_size, lr, val_split=0.1):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    gloss_vocab, sentence_vocab = build_vocabs_from_tsv(tsv_path)
    gloss_vocab.save(config.GLOSS_VOCAB_PATH.replace("gloss_vocab", "gloss_seq_vocab"))
    sentence_vocab.save(config.SENTENCE_VOCAB_PATH)
    print(f"Gloss vocab size: {len(gloss_vocab)}, Sentence vocab size: {len(sentence_vocab)}")

    dataset = Gloss2SentenceDataset(tsv_path, gloss_vocab, sentence_vocab)
    if len(dataset) == 0:
        raise RuntimeError(f"No pairs found in {tsv_path}")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = Gloss2SentenceTransformer(
        gloss_vocab_size=len(gloss_vocab),
        sentence_vocab_size=len(sentence_vocab),
        pad_id=sentence_vocab.pad_id,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=sentence_vocab.pad_id, label_smoothing=0.1)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(config.SEQ2SEQ_CKPT), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            optimizer.zero_grad()
            logits = model(src, tgt_in)  # (batch, T, vocab)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_tok = (tgt_out != sentence_vocab.pad_id).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

        val_loss = evaluate(model, val_loader, criterion, sentence_vocab.pad_id, device)
        train_loss = total_loss / max(1, total_tokens)
        print(f"Epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "gloss_vocab_size": len(gloss_vocab),
                "sentence_vocab_size": len(sentence_vocab),
            }, config.SEQ2SEQ_CKPT)

    print(f"Best val loss: {best_val_loss:.4f}. Checkpoint saved to {config.SEQ2SEQ_CKPT}")


@torch.no_grad()
def evaluate(model, loader, criterion, pad_id, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        n_tok = (tgt_out != pad_id).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(1, total_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/gloss_sentence_pairs.tsv")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_SEQ2SEQ)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.SEQ2SEQ_LEARNING_RATE)
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size, args.lr)
