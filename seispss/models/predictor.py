"""Cross-attention predictor for JEPA latent targets."""

from __future__ import annotations

import torch
import torch.nn as nn

from seispss.models.masking import gather_span_tokens


class PredictorBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        target_queries: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_input = self.norm1(target_queries)
        self_attn_output, _ = self.self_attn(
            self_attn_input,
            self_attn_input,
            self_attn_input,
            need_weights=False,
        )
        target_queries = target_queries + self_attn_output

        cross_attn_input = self.norm2(target_queries)
        cross_attn_output, _ = self.cross_attn(
            cross_attn_input,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        target_queries = target_queries + cross_attn_output
        target_queries = target_queries + self.mlp(self.norm3(target_queries))
        return target_queries


class JEPAPredictor(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        # output_dim defaults to embed_dim (JEPA latent target). The masked-reconstruction
        # ablation reuses the identical predictor body and only swaps the final projection to
        # the raw-patch dimension (in_channels * patch_length), so the two objectives differ
        # solely in the prediction target, not in predictor capacity.
        self.output_dim = int(output_dim) if output_dim is not None else int(embed_dim)
        self.context_proj = nn.Linear(embed_dim, predictor_dim)
        self.pos_proj = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.blocks = nn.ModuleList(
            [
                PredictorBlock(
                    dim=predictor_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, self.output_dim)
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(
        self,
        *,
        context_hidden: torch.Tensor,
        pos_emb: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        context_memory = self.context_proj(context_hidden) + self.pos_proj(pos_emb)
        target_pos = gather_span_tokens(pos_emb, target_indices).flatten(1, 2)
        target_queries = self.mask_token.expand(target_pos.shape[0], target_pos.shape[1], -1)
        target_queries = target_queries + self.pos_proj(target_pos)

        for block in self.blocks:
            target_queries = block(
                target_queries=target_queries,
                memory=context_memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        target_queries = self.norm(target_queries)
        return self.out_proj(target_queries)
