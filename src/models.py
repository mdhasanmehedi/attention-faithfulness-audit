"""
models.py — PhishFormer (faithfulness-audit repository)

This file contains only the PhishFormer architecture used throughout the
faithfulness audit in this repository. It is extracted, unmodified, from
the original project's models.py, which additionally defines five baseline
architectures (CNNOnly, TransformerOnly, LSTMModel, BiLSTMModel,
RandomForestBaseline) used in a separate, companion paper. Those baselines
are intentionally omitted here since they are not used by any script in
this repository, and their checkpoints are not included.

The class below is byte-for-byte identical (module structure, parameter
names, forward-pass logic) to the PhishFormer class in the original
models.py, so that the checkpoints shipped in this repository
(checkpoints/phishformer_seed{seed}_best.pt) load correctly via
utils.load_checkpoint().
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import VOCAB_SIZE, NUM_CLASSES, MAX_LEN, PAD_IDX

# ── Hyperparameters (Section 4.7 of the paper) ───────────────────────────────
EMBED_DIM            = 128   # character embedding dimension
CNN_FILTERS          = 128   # filters per kernel size in CNN component
CNN_KERNELS          = (3, 4, 5)  # n-gram window sizes
TRANSFORMER_HEADS    = 4
TRANSFORMER_LAYERS   = 2
TRANSFORMER_FF_DIM   = 256   # feed-forward dimension inside transformer
DROPOUT              = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# PhishFormer
# ─────────────────────────────────────────────────────────────────────────────

class PhishFormer(nn.Module):
    """
    Hybrid CNN-Transformer for character-level URL classification.

    Key design choice:
      The CNN extracts local n-gram features while PRESERVING the full
      sequence length (200 positions). The resulting position-aware
      feature map is passed directly to the Transformer encoder, so
      self-attention can reason over spatial relationships between
      local patterns — unlike prior hybrids that pool the CNN output
      to a single vector before the Transformer, destroying positional
      information.

    Architecture:
      Embedding(96, 128)
        → [Conv1d(k=3), Conv1d(k=4), Conv1d(k=5)]  each: 128 filters
        → ReLU + concatenate along filter dim  →  (B, 384, 200)
        → transpose  →  (B, 200, 384)           [full-resolution sequence]
        → TransformerEncoder(d_model=384, nhead=4, layers=2)
        → mean pooling over sequence  →  (B, 384)
        → Dropout → Linear(384, 4)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        num_filters: int = CNN_FILTERS,
        kernel_sizes: Tuple[int, ...] = CNN_KERNELS,
        num_heads: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
        ff_dim: int = TRANSFORMER_FF_DIM,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        max_len: int = MAX_LEN,
        pad_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx

        # ── Character embedding ───────────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # ── Parallel CNN towers (one per kernel size) ─────────────────────────
        # Input to Conv1d: (B, embed_dim, seq_len)
        # Output per tower: (B, num_filters, seq_len)  [same-length via padding]
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,   # 'same' padding to preserve sequence length
            )
            for k in kernel_sizes
        ])

        # Total CNN output channels after concatenation along filter dim
        cnn_out_dim = num_filters * len(kernel_sizes)  # 128*3 = 384

        # ── Transformer encoder ───────────────────────────────────────────────
        # d_model must equal cnn_out_dim so we feed CNN output directly
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,    # input/output: (B, seq_len, d_model)
            norm_first=True,     # pre-norm (more stable than post-norm)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # required for stability on macOS MPS/CPU
        )

        # ── Classification head ───────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(cnn_out_dim, num_classes)

        # Store for attention extraction (faithfulness audit)
        self._attention_weights = None
        self._register_attention_hook()

    def _register_attention_hook(self) -> None:
        """
        Register a forward hook on the last TransformerEncoderLayer's
        self-attention module to cache attention weights for the
        faithfulness evaluation (Section 4.6 of the paper).
        """
        last_layer = self.transformer.layers[-1]

        def hook(module, input, output):
            # output is (attn_output, attn_weights) when need_weights=True
            # but TransformerEncoderLayer doesn't expose weights by default.
            # We handle attention extraction explicitly in get_attention_weights().
            pass

        last_layer.register_forward_hook(hook)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: LongTensor of shape (B, seq_len) — character indices
            return_attention: if True, also return attention weights
                              from the last transformer layer

        Returns:
            logits of shape (B, num_classes)
            [optionally: (logits, attention_weights)]
        """
        # Padding mask: True where x == PAD_IDX (Transformer ignores these)
        pad_mask = (x == self.pad_idx)  # (B, seq_len)

        # ── Embedding ─────────────────────────────────────────────────────────
        emb = self.embedding(x)          # (B, seq_len, embed_dim)
        emb = emb.transpose(1, 2)        # (B, embed_dim, seq_len) for Conv1d

        # ── Parallel CNN towers ───────────────────────────────────────────────
        seq_len = emb.size(2)
        conv_outs = []
        for conv in self.conv_layers:
            out = F.relu(conv(emb))          # (B, num_filters, ~seq_len)
            out = out[:, :, :seq_len]        # trim to exact seq_len
            conv_outs.append(out)

        # Concatenate along filter dimension, keep sequence length intact
        cnn_out = torch.cat(conv_outs, dim=1)   # (B, 384, seq_len)
        cnn_out = cnn_out.transpose(1, 2)        # (B, seq_len, 384)

        # ── Transformer encoder ───────────────────────────────────────────────
        if return_attention:
            # Manual pass through layers to extract last-layer attention
            hidden = cnn_out
            attn_weights = None
            for i, layer in enumerate(self.transformer.layers):
                if i == len(self.transformer.layers) - 1:
                    normed = layer.norm1(hidden)
                    attn_out, attn_weights = layer.self_attn(
                        normed, normed, normed,
                        key_padding_mask=pad_mask,
                        need_weights=True,
                        average_attn_weights=True,
                    )
                    hidden = hidden + layer.dropout1(attn_out)
                    hidden = hidden + layer._ff_block(layer.norm2(hidden))
                else:
                    hidden = layer(hidden, src_key_padding_mask=pad_mask)
            transformer_out = hidden
        else:
            transformer_out = self.transformer(
                cnn_out,
                src_key_padding_mask=pad_mask,
            )  # (B, seq_len, 384)

        # ── Mean pooling (ignore padding positions) ───────────────────────────
        # Build a mask to zero out padded positions before averaging
        mask = (~pad_mask).float().unsqueeze(-1)     # (B, seq_len, 1)
        pooled = (transformer_out * mask).sum(dim=1) # (B, 384)
        pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)

        # ── Classification ────────────────────────────────────────────────────
        logits = self.classifier(self.dropout(pooled))   # (B, num_classes)

        if return_attention:
            return logits, attn_weights   # attn_weights: (B, seq_len, seq_len)
        return logits

    def get_attention_weights(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method used by integrated_gradients.py's
        attribution_attention() function (the raw-attention arm of the
        faithfulness audit).
        Returns (logits, per-character importance scores).

        Importance score for position i = mean attention weight received
        by position i across all query positions in the last layer.
        Shape: (B, seq_len).
        """
        logits, attn_weights = self.forward(x, return_attention=True)
        # attn_weights: (B, seq_len, seq_len) — [query, key]
        # Sum over query dimension to get how much each key position
        # was attended to in total
        importance = attn_weights.sum(dim=1)    # (B, seq_len)
        # Normalise per sample to [0, 1]
        importance = importance / importance.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return logits, importance

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils import get_logger, get_device, set_seed, device_info

    set_seed(42)
    device = get_device()
    logger = get_logger()
    logger.info(f"Device: {device_info(device)}")

    B, L = 8, MAX_LEN
    dummy = torch.randint(2, VOCAB_SIZE, (B, L)).to(device)

    model = PhishFormer().to(device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    params = model.count_parameters()
    logger.info(f"PhishFormer | output: {tuple(out.shape)} | params: {params:,}")
    assert out.shape == (B, NUM_CLASSES), "Shape mismatch"
    assert params == 1_791_108, f"Unexpected parameter count: {params:,}"

    with torch.no_grad():
        logits, importance = model.get_attention_weights(dummy)
    logger.info(f"Attention importance shape: {tuple(importance.shape)}")
    assert importance.shape == (B, L), "Attention importance shape mismatch"

    logger.info("models.py OK")
