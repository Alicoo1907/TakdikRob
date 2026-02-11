# gp_sampling.py
import torch
# TODO
def sample_gp(batch_size, seq_len, z_dim, device, lengthscale=0.2, std=1.0):
    """
    Gaussian Process ile zaman boyunca smooth z üretir.
    - Tek bir GP kernel ile tüm kanallar için sampling yapar (hızlı + tutarlı)
    """
    # 1. Zaman grid'i (0'dan 1'e kadar normalize edilmiş)
    t = torch.linspace(0, 1, seq_len, device=device).unsqueeze(1)  # (T, 1)
    dists = (t - t.T).pow(2)  # (T, T) kareli mesafe matrisi

    # 2. RBF kernel matrisi (covariance matrix)
    kernel = torch.exp(-dists / (2 * lengthscale**2))  # (T, T)
    kernel += 1e-5 * torch.eye(seq_len, device=device)  # Numerik stabilite

    # 3. Cholesky çözümü (L)
    L = torch.linalg.cholesky(kernel)  # (T, T)

    # 4. Rastgele latent örnekle (normal dağılım)
    z = torch.randn(batch_size, z_dim, seq_len, device=device)  # (B, Z, T)

    # 5. GP örneği oluştur
    z_gp = torch.matmul(z, L.T)  # (B, Z, T)

    # 6. Biçimi düzelt (B, T, Z)
    z_gp = z_gp.permute(0, 2, 1) * std

    return z_gp