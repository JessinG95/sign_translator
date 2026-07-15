"""
vocab.py
--------
Minimal vocabulary helper shared by the gloss tokens and the target
English sentences. Handles build-from-corpus, save/load, and
encode/decode.
"""

import json
import re
import config


class Vocab:
    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self._add_special_tokens()

    def _add_special_tokens(self):
        for tok in [config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN, config.UNK_TOKEN]:
            self._add(tok)

    def _add(self, token):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    @property
    def pad_id(self):
        return self.token2id[config.PAD_TOKEN]

    @property
    def sos_id(self):
        return self.token2id[config.SOS_TOKEN]

    @property
    def eos_id(self):
        return self.token2id[config.EOS_TOKEN]

    @property
    def unk_id(self):
        return self.token2id[config.UNK_TOKEN]

    def __len__(self):
        return len(self.token2id)

    def build_from_sentences(self, sentences, tokenizer=None):
        tokenizer = tokenizer or word_tokenize
        for sent in sentences:
            for tok in tokenizer(sent):
                self._add(tok)

    def encode(self, tokens, add_sos_eos=False, max_len=None):
        ids = [self.token2id.get(t, self.unk_id) for t in tokens]
        if add_sos_eos:
            ids = [self.sos_id] + ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids, strip_special=True):
        tokens = []
        for i in ids:
            tok = self.id2token.get(int(i), config.UNK_TOKEN)
            if strip_special and tok in (config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN):
                if tok == config.EOS_TOKEN:
                    break
                continue
            tokens.append(tok)
        return tokens

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.token2id, f, indent=2)

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path) as f:
            token2id = json.load(f)
        v.token2id = token2id
        v.id2token = {int(i): t for t, i in token2id.items()}
        return v


def word_tokenize(text: str):
    """Very small, dependency-free tokenizer: lowercase + split on
    word boundaries, keeping simple punctuation as separate tokens."""
    text = text.strip().lower()
    tokens = re.findall(r"[a-z0-9']+|[.,!?]", text)
    return tokens


def detokenize(tokens):
    """Join tokens back into a readable sentence with basic punctuation
    spacing rules."""
    sentence = ""
    for tok in tokens:
        if tok in {".", ",", "!", "?"}:
            sentence = sentence.rstrip() + tok + " "
        else:
            sentence += tok + " "
    sentence = sentence.strip()
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence
