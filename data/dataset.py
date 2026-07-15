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
    def __init__(self, data_dir=config.DATA_DIR, window_size=config.SLIDING_WINDOW_SIZE):
        self.window_size = window_size
        self.samples = []          # list of (filepath, label_str)
        self.classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(cls_dir, fname), cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        arr = np.load(path)  # (T, feature_dim)
        arr = self._fit_to_window(arr)
        label = self.class_to_idx[cls]
        return torch.from_numpy(arr).float(), label, arr.shape[0]

    def _fit_to_window(self, arr):
        """Pad or center-crop a variable-length sequence to a fixed window
        so batches can be stacked. For sequences shorter than the window,
        pads with the last frame; for longer, takes the center slice."""
        T = arr.shape[0]
        w = self.window_size
        if T == w:
            return arr
        if T < w:
            pad = np.repeat(arr[-1:], w - T, axis=0)
            return np.concatenate([arr, pad], axis=0)
        start = (T - w) // 2
        return arr[start:start + w]


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
