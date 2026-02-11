"""
NTU49_7J icin FID (Frechet Inception Distance) hesaplama.

Akis:
  1. Egitilmis encoder'i yukle (train_evaluator.py ile egitilmis)
  2. Gercek veriyi .npy'den oku
  3. Sahte veriyi Results/Full_Train_Generation/ dizininden oku
  4. Encoder ile feature cikart -> FID hesapla
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from scipy import linalg
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# === Parent dizinden importlar ===
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fid import ActFormerEncoder7, frechet_distance

# ============================================================
#  Configuration
# ============================================================
T = 64
V = 7
C = 3
SPLIT = "xsub"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
#  Encoder Yukle
# ============================================================
def load_evaluator(path=None):
    if path is None:
        path = os.path.join(BASE_DIR, "Results", "fid_encoder.pt")

    model = ActFormerEncoder7(T=T, V=V, C=C, out_dim=256)
    try:
        if os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f" Loaded trained FID encoder from {path}")
        else:
            print(f" Warning: Encoder not found at {path}. Using random init!")
    except Exception as e:
        print(f" Error loading encoder: {e}. Using random init!")

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


# ============================================================
#  Gercek ve Sahte Verileri Yukle
# ============================================================
def load_real_and_fake_data(real_data_path, real_label_path, fake_root, max_sequences=None):
    """
    Gercek veri: .npy dosyasindan (N, C=3, T=64, V=7, M=1)
    Sahte veri: Results/Full_Train_Generation/ altindaki .npy dosyalari (3, 7, 64)
    """
    import pickle

    real_tensors, fake_tensors = []  , []

    if not os.path.exists(fake_root):
        print(f" Fake data directory not found: {fake_root}")
        return [], []

    # Gercek veriyi yukle
    real_data = np.load(real_data_path)  # (N, 3, 64, 7, 1)
    with open(real_label_path, 'rb') as f:
        real_names, real_labels = pickle.load(f)

    # Sahte dosyalari listele
    fake_files = os.listdir(fake_root)
    fake_name_map = {}
    for ff in fake_files:
        if ff.endswith('.npy'):
            # Dosya adi: S001C001P001R001A001.skeleton_label_0.npy
            # Name kismini cikar
            name_part = ff.split('_label_')[0] if '_label_' in ff else ff.replace('.npy', '')
            fake_name_map[name_part] = os.path.join(fake_root, ff)

    n_total = len(real_names)
    if max_sequences:
        n_total = min(n_total, max_sequences)

    matched = 0
    for i in tqdm(range(n_total), desc="Loading real+fake pairs"):
        name = real_names[i]
        # .skeleton uzantisini koru veya kaldir
        name_key = name.replace('.skeleton', '') if '.skeleton' in name else name

        # Sahte dosyayi bul
        fake_path = fake_name_map.get(name) or fake_name_map.get(name_key)
        if fake_path is None:
            continue

        # Gercek veri: (3, 64, 7, 1) -> (3, 7, 64)
        real_seq = real_data[i].squeeze(-1)  # (3, 64, 7)
        real_seq = np.transpose(real_seq, (0, 2, 1))  # (3, 7, 64)

        # Sahte veri
        fake_seq = np.load(fake_path)  # (3, 7, 64)

        # Shape kontrol
        min_t = min(real_seq.shape[-1], fake_seq.shape[-1])
        real_seq = real_seq[..., :min_t]
        fake_seq = fake_seq[..., :min_t]

        real_tensors.append(torch.tensor(real_seq, dtype=torch.float32))
        fake_tensors.append(torch.tensor(fake_seq, dtype=torch.float32))
        matched += 1

    print(f"Matched {matched}/{n_total} sequences")
    return real_tensors, fake_tensors


# ============================================================
#  Global FID Hesabi
# ============================================================
def compute_motion_fid(real_tensors, fake_tensors, device="cpu", encoder=None, verbose=True):
    if encoder is None:
        encoder = ActFormerEncoder7(T=T).to(device)
    encoder.eval()

    with torch.no_grad():
        real_feats, fake_feats = [], []
        for seq in tqdm(real_tensors, desc="Encoding real"):
            seq = seq.unsqueeze(0).to(device)
            feat = encoder(seq).cpu().numpy()
            real_feats.append(feat)

        for seq in tqdm(fake_tensors, desc="Encoding fake"):
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
        print(f"Mean Difference (Mean Sq): {diff_sum_sq:.4f}")
        print(f"Covariance Difference (Trace): {trace_part:.4f}")

    return fid_score


# ============================================================
#  Per-Sequence FID Hesabi
# ============================================================
def compute_per_sequence_fid(real_tensors, fake_tensors, device="cpu", encoder=None, names=None):
    if encoder is None:
        encoder = ActFormerEncoder7(T=T).to(device)
    encoder.eval()

    results = []
    for idx in tqdm(range(len(real_tensors)), desc="Per-sequence FID"):
        real_seq = real_tensors[idx].unsqueeze(0).to(device)
        fake_seq = fake_tensors[idx].unsqueeze(0).to(device)

        with torch.no_grad():
            feat_real = encoder(real_seq).cpu().numpy()
            feat_fake = encoder(fake_seq).cpu().numpy()

        mu_r = np.mean(feat_real, axis=0)
        sigma_r = np.cov(feat_real, rowvar=False)
        mu_f = np.mean(feat_fake, axis=0)
        sigma_f = np.cov(feat_fake, rowvar=False)

        fid = frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

        results.append({
            "sequence": names[idx] if names else f"seq_{idx}",
            "FID": float(fid)
        })

    return results


# ============================================================
#  Main
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = os.path.join(BASE_DIR, SPLIT)
    real_data_path = os.path.join(data_dir, "train_data_joint.npy")
    real_label_path = os.path.join(data_dir, "train_label.pkl")
    fake_root = os.path.join(BASE_DIR, "Results", "Full_Train_Generation")

    print(f"Device: {device}")
    print(f"Split: {SPLIT}")
    print(f"Real data: {real_data_path}")
    print(f"Fake data: {fake_root}")

    # Encoder yukle
    encoder = load_evaluator()

    # Verileri yukle
    real_tensors, fake_tensors = load_real_and_fake_data(
        real_data_path, real_label_path, fake_root
    )
    print(f"Loaded {len(real_tensors)} matched sequences.")

    if len(real_tensors) == 0:
        print("No matched sequences found! Make sure test_final.py has been run.")
        return

    # Global FID
    fid_score = compute_motion_fid(real_tensors, fake_tensors, device=device, encoder=encoder)
    print(f"\n Global Motion FID Score: {fid_score:.4f}")

    # Per-sequence FID
    import pickle
    with open(real_label_path, 'rb') as f:
        names, _ = pickle.load(f)

    print("\nCalculating per-sequence FID...")
    results = compute_per_sequence_fid(
        real_tensors, fake_tensors, device=device, encoder=encoder,
        names=names[:len(real_tensors)]
    )

    results_dir = os.path.join(BASE_DIR, "Results")
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(results)
    out_csv = os.path.join(results_dir, "motion_fid_per_sequence.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-sequence FID: {out_csv}")
    if len(df) > 0:
        print(f"Average Motion FID ({len(df)} sequences): {df['FID'].mean():.4f}")


if __name__ == "__main__":
    main()
