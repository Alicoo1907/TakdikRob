import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from scipy import linalg
import pandas as pd
import warnings
import pickle

warnings.filterwarnings("ignore", category=DeprecationWarning)

from net_G import ActFormer_Generator # Consistency
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
            print(f" Warning: Encoder not found at {path}.")
    except Exception as e:
        print(f" Error loading encoder: {e}.")

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


def load_real_and_fake_data(real_data_path, real_label_path, fake_root):
    real_tensors, fake_tensors = [], []

    if not os.path.exists(fake_root):
        print(f" Fake data directory not found: {fake_root}")
        return [], []

    real_data = np.load(real_data_path)  # (N, 3, 64, 7, 1)
    with open(real_label_path, 'rb') as f:
        real_names, _ = pickle.load(f)

    fake_files = os.listdir(fake_root)
    fake_name_map = {}
    for ff in fake_files:
        if ff.endswith('.npy'):
            name_part = ff.split('_label_')[0] if '_label_' in ff else ff.replace('.npy', '')
            fake_name_map[name_part] = os.path.join(fake_root, ff)

    matched = 0
    for i in tqdm(range(len(real_names)), desc="Loading real+fake pairs"):
        name = real_names[i]
        name_key = name.replace('.skeleton', '') if '.skeleton' in name else name

        fake_path = fake_name_map.get(name) or fake_name_map.get(name_key)
        if fake_path is None:
            continue

        real_seq = real_data[i].squeeze(-1)  # (3, 64, 7)
        real_seq = np.transpose(real_seq, (0, 2, 1))  # (3, 7, 64)

        fake_seq = np.load(fake_path)  # (3, 7, 64)

        real_tensors.append(torch.tensor(real_seq, dtype=torch.float32))
        fake_tensors.append(torch.tensor(fake_seq, dtype=torch.float32))
        matched += 1

    print(f"Matched {matched}/{len(real_names)} sequences")
    return real_tensors, fake_tensors


def compute_motion_fid(real_tensors, fake_tensors, device="cpu", encoder=None):
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

    fid_score = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_score


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join(BASE_DIR, SPLIT)
    real_data_path = os.path.join(data_dir, "val_data_joint.npy")
    real_label_path = os.path.join(data_dir, "val_label.pkl")
    fake_root = os.path.join(BASE_DIR, "Results", "Full_Val_Generation")

    print(f"--- Running FID for VALIDATION set ---")
    encoder = load_evaluator()

    real_tensors, fake_tensors = load_real_and_fake_data(real_data_path, real_label_path, fake_root)
    
    if len(real_tensors) == 0:
        print("No matched sequences! Run test_val.py first.")
        return

    fid_score = compute_motion_fid(real_tensors, fake_tensors, device=device, encoder=encoder)
    print(f"\n[Validation] Global Motion FID Score: {fid_score:.4f}")


if __name__ == "__main__":
    main()
