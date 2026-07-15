"""
sign_recognizer.py
--------------------
CNN + BiLSTM model that turns a short window of landmark feature vectors
into a probability distribution over "gloss" classes (i.e. individual
sign / word tokens).

Architecture:
    Input:  (batch, T, FRAME_FEATURE_DIM)
    -> 1D CNN over the time axis (extracts local temporal motion patterns,
       e.g. the shape of a hand movement over a few frames)
    -> BiLSTM (models longer-range temporal dependencies / co-articulation
       between signs)
    -> Attention-weighted pooling over time
    -> Linear classifier -> gloss logits

For continuous signing, the recognizer is applied with a sliding window
(see utils in main.py) so a stream of glosses is produced, which is then
handed to the Seq2Seq transformer to be turned into a grammatical
sentence.
"""

import torch
import torch.nn as nn
import config


class TemporalAttentionPool(nn.Module):
    """Learns to weight time-steps by importance before pooling to a
    single vector -- lets the model focus on the most informative part
    of the gesture window."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (batch, T, dim)
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)  # (batch, T)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)     # (batch, dim)
        return pooled


class SignRecognizer(nn.Module):
    def __init__(self, input_dim=config.FRAME_FEATURE_DIM,
                 num_classes: int = 100,
                 cnn_channels: int = config.RECOGNIZER_CNN_CHANNELS,
                 hidden_dim: int = config.RECOGNIZER_HIDDEN_DIM,
                 lstm_layers: int = config.RECOGNIZER_LSTM_LAYERS,
                 dropout: float = config.RECOGNIZER_DROPOUT):
        super().__init__()

        # 1D CNN feature extractor over time. Padding='same' keeps T fixed.
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.attn_pool = TemporalAttentionPool(hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, lengths=None):
        """
        x: (batch, T, input_dim)
        lengths: optional (batch,) tensor of true sequence lengths for
                 packing (helps with padded batches during training).
        """
        x = x.transpose(1, 2)          # -> (batch, input_dim, T) for Conv1d
        x = self.cnn(x)
        x = x.transpose(1, 2)          # -> (batch, T, cnn_channels)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.bilstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            x, _ = self.bilstm(x)

        pooled = self.attn_pool(x)      # (batch, hidden_dim*2)
        logits = self.classifier(pooled)
        return logits

    @torch.no_grad()
    def predict_gloss_id(self, window: torch.Tensor) -> int:
        """Convenience method for real-time inference on a single window.
        window: (T, input_dim) tensor (no batch dim)."""
        self.eval()
        x = window.unsqueeze(0)  # (1, T, input_dim)
        logits = self.forward(x)
        return int(torch.argmax(logits, dim=-1).item())
    
    @torch.no_grad()
    def predict_with_confidence(self, window: torch.Tensor):
        """Convenience method for real-time inference on a single window.
        window: (T, input_dim) tensor (no batch dim).
        Returns (predicted_class_id, confidence) where confidence is the
        softmax probability of the predicted class."""
        self.eval()
        x = window.unsqueeze(0)  # (1, T, input_dim)
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_id = torch.max(probs, dim=-1)
        return int(pred_id.item()), float(confidence.item())
