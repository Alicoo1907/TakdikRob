import os
import h5py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy import linalg
import pandas as pd
import warnings

from transformer_utils import Block, trunc_normal_, positional_encoding
# TODO
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================
#  ActFormer-style Transformer Feature Extractor (7 joints)
# ============================================================
class ActFormerEncoder7(nn.Module):
    """
    Girdi:  (B, 3, 7, T)
    Çıkış:  (B, 256)  -> FID için 256-dim feature
    """
    def __init__(
        self,
        T=60,
        V=7,
        C=3,
        embed_dim_ratio=16,
        depth=6,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        learnable_pos_embed=True,
        spectral_norm=False,
        out_dim=256
    ):
        super().__init__()
        self.T = T
        self.V = V
        self.C = C
        embed_dim = embed_dim_ratio * V  # ör: 7*16 = 112

        self.input_proj = nn.Linear(C * V, embed_dim)
        if spectral_norm:
            self.input_proj = nn.utils.spectral_norm(self.input_proj)

        if learnable_pos_embed:
            self.temporal_pos_embed = nn.Parameter(torch.zeros(1, T, embed_dim))
            trunc_normal_(self.temporal_pos_embed, std=.02)
        else:
            pe = positional_encoding(embed_dim, T)
            self.temporal_pos_embed = nn.Parameter(pe.unsqueeze(0), requires_grad=False)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=nn.LayerNorm, spectral_norm=spectral_norm
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim)
        if spectral_norm:
            self.head = nn.utils.spectral_norm(self.head)

    def forward(self, x):
        B, C, V, T = x.shape
        assert C == self.C and V == self.V, f"Expected (C,V)=({self.C},{self.V}), got ({C},{V})"
        if T != self.T:
            if T > self.T:
                x = x[..., :self.T]
                T = self.T
            else:
                pad = self.T - T
                pad_tensor = torch.zeros(B, C, V, pad, dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad_tensor], dim=-1)
                T = self.T

        x = x.reshape(B, C * V, T).permute(0, 2, 1).contiguous()
        x = self.input_proj(x)
        x = x + self.temporal_pos_embed[:, :T]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        feat = self.head(x)
        return feat


# ============================================================
#  Fréchet Distance Hesabı (FID formülü)
# ============================================================
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    try:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    except Exception:
        covmean = np.sqrt(sigma1 * sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    if not np.isfinite(covmean).all():
        covmean = np.nan_to_num(covmean)

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    fid = float(np.real(fid))
    return max(fid, 0.0)


# ============================================================
#  Global Motion FID Hesabı
# ============================================================
def compute_motion_fid(real_tensors, fake_tensors, device="cpu", encoder=None, verbose=True):
    if encoder is None:
        encoder = ActFormerEncoder7(T=60).to(device)
    encoder.eval()

    with torch.no_grad():
        real_feats, fake_feats = [], []
        for seq in real_tensors:
            seq = seq.unsqueeze(0).to(device)
            feat = encoder(seq).cpu().numpy()
            real_feats.append(feat)

        for seq in fake_tensors:
            seq = seq.unsqueeze(0).to(device)
            feat = encoder(seq).cpu().numpy()
            fake_feats.append(feat)

    real_feats = np.concatenate(real_feats, axis=0)
    fake_feats = np.concatenate(fake_feats, axis=0)

    mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)

    diff = mu1 - mu2
    diff_sum_sq = diff.dot(diff)
    
    fid_score = frechet_distance(mu1, sigma1, mu2, sigma2)
    trace_part = fid_score - diff_sum_sq

    if verbose:
        print(f"\n--- FID BREAKDOWN ---")
        print(f"Mean Difference (Mean Sq): {diff_sum_sq:.4f} (Düşükse aksiyonlar doğru öğrenilmiştir)")
        print(f"Covariance Difference (Trace): {trace_part:.4f} (Düşükse çeşitlilik/kalite doğrudur)")
    
    return fid_score




# ============================================================
#  HDF5 ve NPY verilerini oku
# ============================================================
def load_real_and_fake_data(h5_path, fake_root, max_sequences=None):
    real_tensors, fake_tensors = [], []
    
    # Klasör yoksa oluştur ve boş dön (hata vermemesi için)
    if not os.path.exists(fake_root):
        print(f" Klasör bulunamadı, oluşturuluyor: {fake_root}")
        os.makedirs(fake_root, exist_ok=True)
        return [], []

    with h5py.File(h5_path, 'r') as f:
        all_keys = list(f.keys())
        if max_sequences:
            all_keys = all_keys[:max_sequences]

        for group_name in tqdm(all_keys, desc="Loading sequences"):
            fake_path = None
            pattern = group_name
            for file in os.listdir(fake_root):
                cleaned = file.replace("(", "").replace(")", "").replace(",", "").replace("'", "")
                if pattern in cleaned:
                    fake_path = os.path.join(fake_root, file)
                    break
            if fake_path is None:
                continue

            group = f[group_name]
            frame_keys = sorted(group.keys(), key=lambda x: int(x))
            T = min(len(frame_keys), 60)
            real_seq = np.zeros((3, 7, T), dtype=np.float32)
            for t, frame_num in enumerate(frame_keys[:T]):
                for j, joint_name in enumerate([
                    'Center', 'ShoulderLeft', 'ElbowLeft', 'WristLeft',
                    'ShoulderRight', 'ElbowRight', 'WristRight'
                ]):
                    real_seq[0, j, t] = group[frame_num][f'{joint_name}/X'][()]
                    real_seq[1, j, t] = group[frame_num][f'{joint_name}/Y'][()]
                    real_seq[2, j, t] = group[frame_num][f'{joint_name}/Z'][()]

            fake_seq = np.load(fake_path)
            if fake_seq.shape[-1] != T:
                min_len = min(fake_seq.shape[-1], T)
                fake_seq = fake_seq[..., :min_len]
                real_seq = real_seq[..., :min_len]

            # real_seq ve fake_seq hangi eksende geliyor emin olalım
            real_seq = torch.tensor(real_seq)
            fake_seq = torch.tensor(fake_seq)

            # Eğer veri (7,3,T) ise, (3,7,T)'e döndür
            if real_seq.shape[0] == 7 and real_seq.shape[1] == 3:
                real_seq = real_seq.permute(1, 0, 2)
            if fake_seq.shape[0] == 7 and fake_seq.shape[1] == 3:
                fake_seq = fake_seq.permute(1, 0, 2)

            real_tensors.append(real_seq)
            fake_tensors.append(fake_seq)


    return real_tensors, fake_tensors


# ============================================================
#  Ana Fonksiyon
# ============================================================
def load_evaluator(path="Results/fid_encoder.pt"):
    # Load trained encoder
    model = ActFormerEncoder7(T=60, V=7, C=3, out_dim=256)
    try:
        if os.path.exists(path):
            state_dict = torch.load(path)
            model.load_state_dict(state_dict)
            print(f" Loaded trained FID encoder from {path}")
        else:
            print(f" Warning: Trained encoder not found at {path}. Using random initialization!")
    except Exception as e:
         print(f" Error loading encoder: {e}. Using random initialization!")
    
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h5_path = "Dataset/HDF5_Dataset_60frame/motions_data60frame.h5"
    fake_root = "Results/Full_Train_Generation"
    
    # Eğitilmiş jüriyi yükle
    encoder = load_evaluator("Results/fid_encoder.pt")

    print(f"Device: {device}")
    print(f"Reading real data from {h5_path}")
    print(f"Reading fake data from {fake_root}")

    real_tensors, fake_tensors = load_real_and_fake_data(h5_path, fake_root)
    print(f"Loaded {len(real_tensors)} sequences.")
    if len(real_tensors) > 0:
        fid_score = compute_motion_fid(real_tensors, fake_tensors, device=device, encoder=encoder)
        print(f"\nGlobal Motion FID Score (Transformer feat): {fid_score:.4f}")

if __name__ == "__main__":
    main()
