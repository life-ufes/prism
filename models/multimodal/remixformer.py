import torch
import torch.nn as nn
from typing import Optional


class DirectedCMFAttention(nn.Module):
    """
    Directed CMF update: g_src (query) attends over tokens_tgt (key,value).
    Follows Eq.(2)–(5) in RemixFormer:
      1. concat [g_src; tokens_tgt]
      2. apply LayerNorm once
      3. split normalized g_tilde, z
      4. compute Q(g_tilde), K(z), V(z)
      5. update g = g_tilde + Linear(Attn(Q,K,V))
    """

    def __init__(
        self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0
    ):
        super().__init__()
        self.ln_seq = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True
        )
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_drop))

    def forward(
        self, g_src: Optional[torch.Tensor], tokens_tgt: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if g_src is None:
            return None
        if tokens_tgt is None:
            return g_src

        # Step 1: concat
        seq = torch.cat([g_src.unsqueeze(1), tokens_tgt], dim=1)  # (B, 1+S, F)

        # Step 2: apply LN once
        seq_norm = self.ln_seq(seq)

        # Step 3: split
        g_tilde = seq_norm[:, :1, :]  # (B,1,F)
        z = seq_norm[:, 1:, :]  # (B,S,F)

        # Step 4: MHA (query=g_tilde, key=z, value=z)
        attn_out, _ = self.attn(query=g_tilde, key=z, value=z)  # (B,1,F)

        # Step 5: residual update
        g_updated = g_tilde + self.out_proj(attn_out)  # (B,1,F)
        return g_updated.squeeze(1)  # (B,F)


class CrossModalityFusion(nn.Module):
    """
    RemixFormer CMF (MICCAI 2022, Sec.2.3).
    - Supports clinical (C), dermoscopic (D), and metadata (M).
    - Returns concat of available globals: [gC,gD,gM].
    """

    def __init__(
        self,
        dim: int,
        meta_in_dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_in_dim, dim), nn.BatchNorm1d(dim), nn.ReLU()
        )

        # Directed attention blocks
        self.attn = nn.ModuleDict(
            {
                "C_from_D": DirectedCMFAttention(dim, num_heads, attn_drop, proj_drop),
                "D_from_C": DirectedCMFAttention(dim, num_heads, attn_drop, proj_drop),
                "M_from_C": DirectedCMFAttention(dim, num_heads, attn_drop, proj_drop),
                "M_from_D": DirectedCMFAttention(dim, num_heads, attn_drop, proj_drop),
            }
        )

    def forward(
        self,
        local_C: Optional[torch.Tensor],
        local_D: Optional[torch.Tensor],
        meta_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:

        assert (
            local_C is None or local_C.dim() == 4
        ), "local_C must be a 4D tensor or None"
        assert (
            local_D is None or local_D.dim() == 4
        ), "local_D must be a 4D tensor or None"
        assert (
            meta_vec is None or meta_vec.dim() == 2
        ), "meta_vec must be a 2D tensor or None"

        # local features
        lC = self._to_tokens(local_C)  # transforms (B, C, H, W) -> (B, H*W, C)
        lD = self._to_tokens(local_D)

        # global features
        gC = self._global_average_pool(local_C)
        gD = self._global_average_pool(local_D)
        gM = self.meta_proj(meta_vec) if meta_vec is not None else None

        # Image-image exchange
        gC = self.attn["C_from_D"](gC, lD)
        gD = self.attn["D_from_C"](gD, lC)

        # Metadata fusion
        gM_C = self.attn["M_from_C"](gM, lC) if gM is not None else None
        gM_D = self.attn["M_from_D"](gM, lD) if gM is not None else None

        # Concatenate available globals
        outs = [x for x in [gC, gD, gM_C, gM_D] if x is not None]
        fused = torch.concat(outs, dim=-1) if outs else torch.zeros(0)
        return fused

    def _to_tokens(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return x.flatten(2).transpose(1, 2) if x is not None else None

    def _global_average_pool(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return self.pool(x).flatten(1) if x is not None else None


class CrossModalityFusionAdapter(nn.Module):
    def __init__(self, vision_backbone_ouput_size, n_metadata):
        super().__init__()
        self.feature_fusion = CrossModalityFusion(
            vision_backbone_ouput_size, num_heads=8, meta_in_dim=n_metadata
        )

    def forward(self, cnn_features, metadata):
        return self.feature_fusion(
            local_C=cnn_features,
            local_D=None,  # The forward pass of CrossModalityFusion handles None for the second image modality.
            meta_vec=metadata.float(),
        )
