"""
NTU49_7J - ACC (Recognition Accuracy on Generated Motions)

classifier_full.pt ile fake hareketlerin taninma oranini hesaplar.

Kullanim:
  python acc.py

Gereksinimler:
  - Results/classifier_full.pt  (train_evaluator.py)
  - Results/Full_Val_Generation/  (test_val.py)
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from net_G import ActFormer_Generator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fid import ActFormerEncoder7

# ============================================================
T = 64
V = 7
C = 3
NUM_CLASSES = 49
SPLIT = "xsub"
BATCH_SIZE = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActionClassifier(nn.Module):
    def __init__(self, num_classes=49):
        super().__init__()
        self.encoder = ActFormerEncoder7(T=T, V=7, C=3, out_dim=256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        return self.fc(feat)


def load_classifier():
    model = ActionClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    path = os.path.join(BASE_DIR, "Results", "classifier_full.pt")

    if not os.path.exists(path):
        print(f"Error: {path} not found! Run train_evaluator.py first.")
        return None

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    print(f"Loaded classifier from {path}")
    model.eval()
    return model


def load_fake_data(fake_root):
    """Fake dosyalari yukler ve kontroller yapar."""
    files = sorted([f for f in os.listdir(fake_root) if f.endswith('.npy') and '_label_' in f])

    if not files:
        print("Error: No fake .npy files found!")
        return [], []

    fake_data, fake_labels = [], []
    print(f"Loading {len(files)} fake sequences...")

    for ff in tqdm(files, desc="Loading fake data"):
        label = int(ff.split('_label_')[1].replace('.npy', ''))
        seq = np.load(os.path.join(fake_root, ff))
        fake_data.append(torch.tensor(seq, dtype=torch.float32))
        fake_labels.append(label)

    # === Fake Data Kontrol ===
    print(f"\n--- Fake Data Kontrolu ---")
    print(f"  Toplam: {len(fake_data)} sequence")
    print(f"  Shape: {fake_data[0].shape}")

    from collections import Counter
    dist = Counter(fake_labels)
    print(f"  Sinif sayisi: {len(dist)}")

    # NaN / Inf kontrolu
    nan_count = sum(1 for d in fake_data if torch.isnan(d).any())
    inf_count = sum(1 for d in fake_data if torch.isinf(d).any())
    if nan_count > 0:
        print(f"  WARNING: {nan_count} sequence NaN iceriyor!")
    if inf_count > 0:
        print(f"  WARNING: {inf_count} sequence Inf iceriyor!")
    if nan_count == 0 and inf_count == 0:
        print(f"  NaN/Inf: Yok (OK)")

    all_vals = torch.cat([d.flatten() for d in fake_data])
    print(f"  Deger araligi: [{all_vals.min():.4f}, {all_vals.max():.4f}]")
    print(f"  Mean: {all_vals.mean():.4f}, Std: {all_vals.std():.4f}")
    print()

    return fake_data, fake_labels


def compute_acc(fake_data, fake_labels, model):
    correct, total = 0, 0
    per_class = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(fake_data), BATCH_SIZE), desc="Computing ACC"):
            end = min(i + BATCH_SIZE, len(fake_data))
            batch = torch.stack(fake_data[i:end]).to(DEVICE)
            labels = torch.tensor(fake_labels[i:end], dtype=torch.long).to(DEVICE)
            preds = torch.argmax(model(batch), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                l = int(l)
                per_class.setdefault(l, {"c": 0, "t": 0})
                per_class[l]["t"] += 1
                if p == l:
                    per_class[l]["c"] += 1

    acc = correct / total if total else 0.0
    per_class_acc = {c: d["c"]/d["t"] for c, d in sorted(per_class.items())}
    return acc, per_class_acc


def main():
    fake_root = os.path.join(BASE_DIR, "Results", "Full_Val_Generation")
    print(f"=== NTU49_7J ACC (Recognition Accuracy) ===")
    print(f"Split: {SPLIT} | Classes: {NUM_CLASSES} | Device: {DEVICE}\n")

    if not os.path.exists(fake_root):
        print(f"Error: {fake_root} not found. Run test_val.py first!")
        return

    model = load_classifier()
    if model is None:
        return

    fake_data, fake_labels = load_fake_data(fake_root)
    if not fake_data:
        return

    acc, per_class_acc = compute_acc(fake_data, fake_labels, model)

    print(f"\n{'Class':>6}  {'ACC':>8}")
    print("-" * 18)
    for c in sorted(per_class_acc):
        print(f"  {c:>3d}   {per_class_acc[c]:.4f}")
    print("-" * 18)
    print(f"\nOverall ACC: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Mean per-class ACC: {np.mean(list(per_class_acc.values())):.4f}")


if __name__ == "__main__":
    main()
