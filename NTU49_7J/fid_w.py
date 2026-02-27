"""
NTU49_7J - FID_w (Action-Wise FID) Hesabi

Her aksiyon sinifi icin ayri FID hesaplayip ortalamasini alir.
FID_w = (1/K) * sum(FID_k)  where K = number of classes
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from scipy import linalg
import pickle
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from net_G import ActFormer_Generator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fid import ActFormerEncoder7, frechet_distance

# ============================================================
#  Configuration
# ============================================================
T = 64
V = 7
C = 3
SPLIT = "xsub"
NUM_CLASSES = 49
MIN_SAMPLES_PER_CLASS = 2  # FID icin en az 2 sample gerekli

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_evaluator(path=None):
    if path is None:
        path = os.path.join(BASE_DIR, "Results", "fid_encoder.pt")

    model = ActFormerEncoder7(T=T, V=V, C=C, out_dim=256)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded FID encoder from {path}")
    else:
        print(f"Warning: Encoder not found at {path}")

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


def load_real_and_fake_by_class(real_data_path, real_label_path, fake_root):
    """
    Returns: dict[class_id] -> {"real": [tensors], "fake": [tensors]}
    """
    class_data = {c: {"real": [], "fake": []} for c in range(NUM_CLASSES)}

    real_data = np.load(real_data_path)  # (N, 3, 64, 7, 1)
    with open(real_label_path, 'rb') as f:
        real_names, real_labels = pickle.load(f)

    fake_files = os.listdir(fake_root)
    fake_name_map = {}
    for ff in fake_files:
        if ff.endswith('.npy'):
            name_part = ff.split('_label_')[0] if '_label_' in ff else ff.replace('.npy', '')
            fake_name_map[name_part] = os.path.join(fake_root, ff)

    matched = 0
    for i in tqdm(range(len(real_names)), desc="Loading real+fake pairs by class"):
        name = real_names[i]
        label = int(real_labels[i])
        name_key = name.replace('.skeleton', '') if '.skeleton' in name else name

        fake_path = fake_name_map.get(name) or fake_name_map.get(name_key)
        if fake_path is None:
            continue

        real_seq = real_data[i].squeeze(-1)       # (3, 64, 7)
        real_seq = np.transpose(real_seq, (0, 2, 1))  # (3, 7, 64)
        fake_seq = np.load(fake_path)              # (3, 7, 64)

        class_data[label]["real"].append(torch.tensor(real_seq, dtype=torch.float32))
        class_data[label]["fake"].append(torch.tensor(fake_seq, dtype=torch.float32))
        matched += 1

    print(f"Matched {matched}/{len(real_names)} sequences")
    return class_data


def encode_sequences(tensors, encoder, device):
    feats = []
    with torch.no_grad():
        for seq in tensors:
            seq = seq.unsqueeze(0).to(device)
            feat = encoder(seq).cpu().numpy()
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def compute_fid_w(class_data, encoder, device):
    fid_per_class = {}
    skipped = []

    for class_id in tqdm(range(NUM_CLASSES), desc="Computing FID per class"):
        real_list = class_data[class_id]["real"]
        fake_list = class_data[class_id]["fake"]

        if len(real_list) < MIN_SAMPLES_PER_CLASS or len(fake_list) < MIN_SAMPLES_PER_CLASS:
            skipped.append(class_id)
            continue

        real_feats = encode_sequences(real_list, encoder, device)
        fake_feats = encode_sequences(fake_list, encoder, device)

        mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
        mu2, sigma2 = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)

        fid = frechet_distance(mu1, sigma1, mu2, sigma2)
        fid_per_class[class_id] = fid

    if skipped:
        print(f"\nSkipped {len(skipped)} classes (< {MIN_SAMPLES_PER_CLASS} samples): {skipped}")

    return fid_per_class


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join(BASE_DIR, SPLIT)
    real_data_path = os.path.join(data_dir, "val_data_joint.npy")
    real_label_path = os.path.join(data_dir, "val_label.pkl")
    fake_root = os.path.join(BASE_DIR, "Results", "Full_Val_Generation")

    print(f"=== NTU49_7J FID_w (Action-Wise FID) ===")
    print(f"Split: {SPLIT} | Classes: {NUM_CLASSES}")
    print(f"Device: {device}\n")

    encoder = load_evaluator()
    class_data = load_real_and_fake_by_class(real_data_path, real_label_path, fake_root)

    # Sinif basina sample sayilarini goster
    print("\nSamples per class:")
    for c in range(NUM_CLASSES):
        n = len(class_data[c]["real"])
        if n > 0:
            print(f"  Class {c:2d}: {n} pairs")

    fid_per_class = compute_fid_w(class_data, encoder, device)

    if fid_per_class:
        print(f"\n{'='*50}")
        print(f"{'Class':>8}  {'FID':>12}  {'Samples':>8}")
        print(f"{'='*50}")
        for c in sorted(fid_per_class.keys()):
            n = len(class_data[c]["real"])
            print(f"  {c:>5d}  {fid_per_class[c]:>12.4f}  {n:>8d}")

        fid_w = np.mean(list(fid_per_class.values()))
        print(f"{'='*50}")
        print(f"\n[Validation] FID_w (Action-Wise): {fid_w:.4f}")
        print(f"  Computed over {len(fid_per_class)}/{NUM_CLASSES} classes")
    else:
        print("No class had enough samples to compute FID!")


if __name__ == "__main__":
    main()
