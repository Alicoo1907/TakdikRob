# net_G.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from transformer_utils import Block, trunc_normal_, positional_encoding


class ActFormer_Generator(nn.Module):
    """
    ActFormer-based Generator for motion generation

    Z: latent dimension (noise)
    T: sequence length (fixed to 60)
    C: coordinate dimension (usually 3 for X,Y,Z)
    V: number of joints (7 in your case)

    This version adds a learnable temporal Conv1d over the GP latent sequence:
      z_gp (B, T, Z) --[Conv1d over time]--> z_filt (B, T, Z) --> Linear --> Transformer
    """

    def __init__(self,
                 Z=64,
                 T=60,
                 C=3,
                 V=7,
                 spectral_norm=True,
                 out_normalize=None,
                 learnable_pos_embed=True,
                 embed_dim_ratio=32,
                 depth=8,
                 num_heads=8,
                 mlp_ratio=2.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.2,
                 norm_layer=None,
                 num_class=15):
        super().__init__()

        self.Z = Z
        self.T = T
        self.C = C
        self.V = V
        self.spectral_norm = spectral_norm
        self.out_normalize = out_normalize
        embed_dim = embed_dim_ratio * V

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # === NEW: Learnable temporal filter over GP latent sequence ===
        # Conv1d: channels=Z, length=T  -> (B, Z, T) in/out
        self.temporal_filter = nn.Conv1d(Z, Z, kernel_size=3, padding=1)
        if spectral_norm:
            self.temporal_filter = nn.utils.spectral_norm(self.temporal_filter)
        # Nonlinearity for filtered latent (kept lightweight)
        # TODO : burası tanh'dı
        self.temporal_act = nn.GELU()

        # Frame-wise input embedding
        self.input_embedding = nn.Linear(Z, embed_dim)
        if spectral_norm:
            self.input_embedding = nn.utils.spectral_norm(self.input_embedding)

        # Class label condition embedding
        self.class_embedding = nn.Embedding(num_class, embed_dim)
        nn.init.orthogonal_(self.class_embedding.weight)

        # Positional encoding
        if learnable_pos_embed:
            self.temporal_pos_embed = nn.Parameter(torch.zeros(1, T+1, embed_dim))
            trunc_normal_(self.temporal_pos_embed, std=.02)
        else:
            temporal_pos_embed = positional_encoding(embed_dim, T)
            class_pos_embed = torch.zeros(1, embed_dim)
            self.temporal_pos_embed = nn.Parameter(torch.cat((class_pos_embed, temporal_pos_embed), 0).unsqueeze(0))
            self.temporal_pos_embed.requires_grad_(False)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                spectral_norm=spectral_norm)
            for i in range(depth)
        ])

        self.temporal_norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, C * V)
        if spectral_norm:
            self.head = nn.utils.spectral_norm(self.head)


    def forward(self, z, y):
        """
        z can be:
          - (B, Z)            -> will be expanded to (B, T, Z)
          - (B, T, Z)         -> used as-is
          - (B, Z, T)         -> will be transposed to (B, T, Z)

        Returns:
          x: (B, C, V, T)
        """
        # Ensure shape (B, T, Z)
        if len(z.shape) == 2:
            z = z.unsqueeze(1).repeat(1, self.T, 1)  # (B, T, Z)
        elif len(z.shape) == 3:
            if z.shape[1] == self.Z and z.shape[2] == self.T:
                z = z.transpose(1, 2)  # (B, T, Z)
            # else already (B, T, Z) assumed
        else:
            raise ValueError(f"Unexpected shape for z: {z.shape}")

        # === NEW: Learnable temporal filtering over the GP latent ===
        # Conv1d expects (B, C=Z, L=T)
        z = self.temporal_filter(z.transpose(1, 2)).transpose(1, 2)  # (B, T, Z)
        z = self.temporal_act(z)

        # Input projection
        x = self.input_embedding(z)  # (B, T, embed_dim)

        # Class conditioning as a prefix token
        y = self.class_embedding(y).unsqueeze(1)  # (B, 1, embed_dim)
        x = torch.cat((y, x), 1)  # (B, T+1, embed_dim)

        # Positional embeddings
        x = x + self.temporal_pos_embed

        # Transformer blocks
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        # Output projection
        x = self.temporal_norm(x)
        x = self.head(x)  # (B, T+1, C*V)

        # Remove class token and reshape to (B, C, V, T)
        x = x[:, 1:].view(x.shape[0], self.T, self.V, self.C)  # (B, T, V, C)
        x = x.permute(0, 3, 2, 1)  # (B, C, V, T)

        return x
