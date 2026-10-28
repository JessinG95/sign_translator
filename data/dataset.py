"""
dataset.py
-----------
Two PyTorch Datasets:

1. SignRecognitionDataset
   Loads landmark-sequence .npy files (one per recorded sign example,
   produced by data/collect_data.py) and their gloss label, for training
   the CNN+BiLSTM sign recognizer.

   Expected directory layout under config.DATA_DIR:
       dataset/
         GLOSS_WORD_1/
             sample_0001.npy
             sample_0002.npy
         GLOSS_WORD_2/
             sample_0001.npy
         ...

2. Gloss2SentenceDataset
   Loads a simple CSV/TSV of (gloss_sequence, target_sentence) pairs for
   training the seq2seq Transformer, e.g.:
       i school go yesterday <TAB> I went to school yesterday.
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from utils.vocab import word_tokenize


class SignRecognitionDataset(Dataset):
    def __init__(
        self,
        data_dir=config.DATA_DIR,
        window_size=config.SLIDING_WINDOW_SIZE,
        classes=None,
    ):
        self.window_size = window_size
        self.data_dir = data_dir
        self.samples = []

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {data_dir}"
            )

        # Use a shared class ordering when one is supplied.
        if classes is None:
            self.classes = sorted([
                d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ])
        else:
            self.classes = list(classes)

        if not self.classes:
            raise RuntimeError(
                f"No class folders found in {data_dir}"
            )

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }

        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)

            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Expected class folder not found: {class_dir}"
                )

            for filename in sorted(os.listdir(class_dir)):
                if filename.endswith(".npy"):
                    self.samples.append(
                        (os.path.join(class_dir, filename), class_name)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_name = self.samples[idx]

        arr = np.load(path)

        if arr.ndim != 2:
            raise ValueError(
                f"Expected a 2D array (frames, features): {path}"
            )

        if arr.shape[1] != config.FRAME_FEATURE_DIM:
            raise ValueError(
                f"Expected {config.FRAME_FEATURE_DIM} features, "
                f"but found {arr.shape[1]} in {path}"
            )

        if arr.shape[0] == 0:
            raise ValueError(f"Empty sequence found: {path}")

        if not np.isfinite(arr).all():
            raise ValueError(f"NaN or infinite values found: {path}")

        arr = self._fit_to_window(arr)
        label = self.class_to_idx[class_name]

        return torch.from_numpy(arr).float(), label, arr.shape[0]

    def _fit_to_window(self, arr):
        """Center-crop or pad a sequence to the configured window size."""
        T = arr.shape[0]
        window = self.window_size

        if T == window:
            return arr

        if T < window:
            pad = np.repeat(arr[-1:], window - T, axis=0)
            return np.concatenate([arr, pad], axis=0)

        start = (T - window) // 2
        return arr[start:start + window]

class Gloss2SentenceDataset(Dataset):
    def __init__(self, tsv_path, gloss_vocab, sentence_vocab,
                 max_gloss_len=config.MAX_GLOSS_LEN,
                 max_sentence_len=config.MAX_SENTENCE_LEN):
        self.pairs = []
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) != 2:
                    continue
                gloss_str, sentence_str = row
                self.pairs.append((gloss_str.strip(), sentence_str.strip()))

        self.gloss_vocab = gloss_vocab
        self.sentence_vocab = sentence_vocab
        self.max_gloss_len = max_gloss_len
        self.max_sentence_len = max_sentence_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gloss_str, sentence_str = self.pairs[idx]
        gloss_tokens = gloss_str.lower().split()
        sentence_tokens = word_tokenize(sentence_str)

        src = self.gloss_vocab.encode(gloss_tokens, max_len=self.max_gloss_len)

        # Decoder input: <sos> + sentence (teacher forcing)
        # Decoder target: sentence + <eos>
        full = self.sentence_vocab.encode(
            sentence_tokens, add_sos_eos=True, max_len=self.max_sentence_len + 1)
        tgt_in = full[:-1]
        tgt_out = full[1:]

        return (torch.tensor(src, dtype=torch.long),
                torch.tensor(tgt_in, dtype=torch.long),
                torch.tensor(tgt_out, dtype=torch.long))


def build_vocabs_from_tsv(tsv_path):
    """Helper to build both vocabularies from a training TSV in one pass."""
    from utils.vocab import Vocab
    gloss_vocab, sentence_vocab = Vocab(), Vocab()
    gloss_sentences, sentences = [], []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                continue
            gloss_str, sentence_str = row
            gloss_sentences.append(gloss_str)
            sentences.append(sentence_str)

    gloss_vocab.build_from_sentences(gloss_sentences, tokenizer=lambda s: s.lower().split())
    sentence_vocab.build_from_sentences(sentences, tokenizer=word_tokenize)
    return gloss_vocab, sentence_vocab
