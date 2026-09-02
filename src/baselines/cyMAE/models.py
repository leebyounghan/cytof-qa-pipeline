"""Model factories: cyMAE encoder/MAE wrappers parameterized by per-cohort num_features,
plus the per-step classification head used on top of the frozen encoder.

cyMAE upstream factories (vendor.modeling_pretrain.pretrain_mae_30D_6L etc.) hardcode
`encoder_num_features=30` (the MDIPA panel size). We need cohort-specific marker counts
so we instantiate `PretrainVisionTransformer` directly here with `num_features=M_c` while
keeping every other hyperparameter identical to the released `_30D_6L` variant.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from .vendor.modeling_pretrain import PretrainVisionTransformer, PretrainVisionTransformerEncoder


# Hyperparameters fixed across cohorts; match the released `pretrain_mae_30D_6L` variant.
ENCODER_EMBED_DIM = 30
ENCODER_DEPTH = 6
ENCODER_HEADS = 6
DECODER_EMBED_DIM = 15
DECODER_DEPTH = 2
DECODER_HEADS = 3
MLP_RATIO = 4


def build_mae(num_features: int) -> PretrainVisionTransformer:
    """Instantiate the cyMAE pretrain model for a cohort with `num_features` channels.

    Architecture is identical to the released `pretrain_mae_30D_6L` checkpoint, only
    `encoder_num_features` (and the decoder's matching `num_features`) differ.
    """
    return PretrainVisionTransformer(
        encoder_num_features=num_features,
        encoder_embed_dim=ENCODER_EMBED_DIM,
        encoder_depth=ENCODER_DEPTH,
        encoder_num_heads=ENCODER_HEADS,
        encoder_num_classes=0,
        decoder_num_classes=1,
        decoder_embed_dim=DECODER_EMBED_DIM,
        decoder_depth=DECODER_DEPTH,
        decoder_num_heads=DECODER_HEADS,
        mlp_ratio=MLP_RATIO,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def build_encoder(num_features: int) -> PretrainVisionTransformerEncoder:
    """Standalone encoder for downstream embedding extraction (frozen at train-heads /
    inference time). Same hyperparameters as `build_mae` produces."""
    return PretrainVisionTransformerEncoder(
        num_features=num_features,
        num_classes=0,
        embed_dim=ENCODER_EMBED_DIM,
        depth=ENCODER_DEPTH,
        num_heads=ENCODER_HEADS,
        mlp_ratio=MLP_RATIO,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
    )


@torch.no_grad()
def embed_cells(
    encoder: PretrainVisionTransformerEncoder,
    x: torch.Tensor,
    batch_size: int = 16384,
    return_device: torch.device | str | None = None,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
) -> torch.Tensor:
    """Push (N, M) cell tensor through the encoder with no masking, mean-pool the
    M-token output → (N, embed_dim). Used by train_heads.py and inference.py.

    return_device=None keeps the result on the encoder's device (avoids a CPU
    roundtrip when the caller will push it back to GPU). Pass torch.device("cpu")
    for the legacy behavior.
    """
    encoder.eval()
    device = next(encoder.parameters()).device
    out: list[torch.Tensor] = []
    n = x.shape[0]
    use_amp = autocast_dtype is not None and device.type == "cuda"
    mask = None
    for start in range(0, n, batch_size):
        chunk = x[start : start + batch_size].to(device, non_blocking=True)
        if mask is None or mask.shape != chunk.shape:
            mask = torch.zeros(chunk.shape, dtype=torch.bool, device=device)
        if use_amp:
            with torch.autocast("cuda", dtype=autocast_dtype):
                feat = encoder.forward_features(chunk, mask)
        else:
            feat = encoder.forward_features(chunk, mask)
        out.append(feat.mean(dim=1).float())
    emb = torch.cat(out, dim=0)
    if return_device is not None:
        emb = emb.to(return_device)
    return emb


class StepHead(nn.Module):
    """Per-(cohort, internal-node) classification head on top of mean-pooled embeddings.

    Architecture: Linear(embed_dim → 64) → ReLU → Linear(64 → num_classes).
    `num_classes = len(ti.categories)` — categories already include "Unassigned" per
    src/preprocess.py, so the head does not need a separate "+1" slot.
    """

    def __init__(self, embed_dim: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
