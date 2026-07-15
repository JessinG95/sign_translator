"""
seq2seq_transformer.py
------------------------
A compact Transformer encoder-decoder that maps a sequence of recognized
"gloss" tokens (raw sign order, e.g. ["I", "SCHOOL", "GO", "YESTERDAY"])
to a grammatically correct natural-language sentence
(e.g. "I went to school yesterday.").

This is intentionally a small, from-scratch Transformer (not a huge
pretrained LM) so it:
    - trains fast on a modest paired gloss/sentence dataset
    - runs comfortably in real time on CPU or a modest GPU
    - has no external download / API dependency

If you have a larger paired dataset, config.py's dimensions can be scaled
up, or this module can be swapped for a fine-tuned T5-small
(see `T5Seq2Seq` wrapper at the bottom for that optional path).
"""

import math
import torch
import torch.nn as nn
import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                              (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Gloss2SentenceTransformer(nn.Module):
    def __init__(self, gloss_vocab_size, sentence_vocab_size,
                 d_model=config.D_MODEL, nhead=config.NHEAD,
                 num_encoder_layers=config.NUM_ENCODER_LAYERS,
                 num_decoder_layers=config.NUM_DECODER_LAYERS,
                 dim_feedforward=config.FFN_DIM,
                 dropout=config.SEQ2SEQ_DROPOUT,
                 pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embed = nn.Embedding(gloss_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(sentence_vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(d_model, sentence_vocab_size)

    def _make_pad_mask(self, seq):
        return seq == self.pad_id  # (batch, T) True where padded

    @staticmethod
    def _make_causal_mask(size, device):
        return nn.Transformer.generate_square_subsequent_mask(size).to(device)

    def forward(self, src, tgt_in):
        """
        src:    (batch, S) gloss token ids
        tgt_in: (batch, T) sentence token ids, shifted right (starts with <sos>)
        returns logits: (batch, T, sentence_vocab_size)
        """
        device = src.device
        src_key_padding_mask = self._make_pad_mask(src)
        tgt_key_padding_mask = self._make_pad_mask(tgt_in)
        tgt_mask = self._make_causal_mask(tgt_in.size(1), device)

        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt_in) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.generator(out)

    @torch.no_grad()
    def greedy_decode(self, src, sos_id, eos_id, max_len=config.MAX_SENTENCE_LEN):
        """Autoregressive greedy decoding for real-time inference.
        src: (1, S) gloss token ids for a single utterance."""
        self.eval()
        device = src.device
        src_key_padding_mask = self._make_pad_mask(src)
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask)

        ys = torch.tensor([[sos_id]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_emb = self.pos_enc(self.tgt_embed(ys) * math.sqrt(self.d_model))
            tgt_mask = self._make_causal_mask(ys.size(1), device)
            out = self.transformer.decoder(
                tgt_emb, memory, tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask)
            logits = self.generator(out[:, -1, :])
            next_id = int(torch.argmax(logits, dim=-1).item())
            ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == eos_id:
                break
        return ys.squeeze(0).tolist()


# ---------------------------------------------------------------------
# Optional: swap in a pretrained T5-small instead of the from-scratch
# Transformer above, if you have `transformers` installed and prefer to
# fine-tune a pretrained language model for stronger grammar generalization
# on small datasets. Not used by default.
# ---------------------------------------------------------------------
class T5Seq2Seq:
    """Thin optional wrapper around a fine-tuned T5-small model.
    Only import `transformers` if this class is actually used, so the
    core system has no hard dependency on it.
    """

    def __init__(self, model_name_or_path="t5-small", device="cpu"):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
        self.device = device

    def translate(self, gloss_tokens):
        text = "translate gloss to sentence: " + " ".join(gloss_tokens)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(**inputs, max_length=config.MAX_SENTENCE_LEN)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
