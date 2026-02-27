"""
NAO (7 Joint) - ACC (Recognition Accuracy on Generated Motions)

classifier_full.pt ile fake hareketlerin taninma oranini hesaplar.

Kullanim:
  python acc.py

Gereksinimler:
  - Results/classifier_full.pt  (train_evaluator.py)
  - Results/K18_Generation  (test_final.py)
"""
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from fid import ActFormerEncoder7

# ============================================================
NUM_CLASSES = 15
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActionClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.encoder = ActFormerEncoder7(T=60, V=7, C=3, out_dim=256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        return self.fc(feat)


def load_classifier():
    model = ActionClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    path = "Results/classifier_full.pt"

    if not os.path.exists(path):
        print(f"Error: {path} not found! Run train_evaluator.py first.")
        return None

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    print(f"Loaded classifier from {path}")
    model.eval()
    return model


def load_fake_data(fake_root):
    """Fake dosyalari yukler ve basit kontroller yapar."""
    fake_data, fake_labels, fake_names = [], [], []

    files = sorted([f for f in os.listdir(fake_root) if f.endswith('.npy') and '_label_' in f])

    if not files:
        print("Error: No fake .npy files found!")
        return [], [], []

    for ff in files:
        label = int(ff.split('_label_')[1].replace('.npy', ''))
        seq = np.load(os.path.join(fake_root, ff))
        fake_t = torch.tensor(seq, dtype=torch.float32)

        if fake_t.shape[0] == 7 and fake_t.shape[1] == 3:
            fake_t = fake_t.permute(1, 0, 2)

        fake_data.append(fake_t)
        fake_labels.append(label)
        fake_names.append(ff)

    # === Fake Data Kontrol ===
    print(f"\n--- Fake Data Kontrolu ---")
    print(f"  Toplam: {len(fake_data)} sequence")
    print(f"  Shape: {fake_data[0].shape}")
    print(f"  Sinif dagilimi:")
    from collections import Counter
    dist = Counter(fake_labels)
    for c in sorted(dist.keys()):
        print(f"    Class {c:2d}: {dist[c]} sample")

    # NaN / Inf kontrolu
    nan_count = sum(1 for d in fake_data if torch.isnan(d).any())
    inf_count = sum(1 for d in fake_data if torch.isinf(d).any())
    if nan_count > 0:
        print(f"  WARNING: {nan_count} sequence NaN iceriyor!")
    if inf_count > 0:
        print(f"  WARNING: {inf_count} sequence Inf iceriyor!")
    if nan_count == 0 and inf_count == 0:
        print(f"  NaN/Inf: Yok (OK)")

    # Deger araligi
    all_vals = torch.cat([d.flatten() for d in fake_data])
    print(f"  Deger araligi: [{all_vals.min():.4f}, {all_vals.max():.4f}]")
    print(f"  Mean: {all_vals.mean():.4f}, Std: {all_vals.std():.4f}")
    print()

    return fake_data, fake_labels, fake_names


def compute_acc(fake_data, fake_labels, model):
    correct, total = 0, 0
    per_class = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(fake_data), BATCH_SIZE), desc="ACC"):
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
    fake_root = "Results/K18_Generation"
    print(f"=== NAO ACC (Recognition Accuracy) ===")
    print(f"Classes: {NUM_CLASSES} | Device: {DEVICE}")

    if not os.path.exists(fake_root):
        print(f"Error: {fake_root} not found. Run test_final.py first!")
        return

    model = load_classifier()
    if model is None:
        return

    fake_data, fake_labels, fake_names = load_fake_data(fake_root)
    if not fake_data:
        return

    acc, per_class_acc = compute_acc(fake_data, fake_labels, model)

    print(f"{'Class':>6}  {'ACC':>8}")
    print("-" * 18)
    for c in sorted(per_class_acc):
        print(f"  {c:>3d}   {per_class_acc[c]:.4f}")
    print("-" * 18)
    print(f"\nOverall ACC: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Mean per-class ACC: {np.mean(list(per_class_acc.values())):.4f}")


if __name__ == "__main__":
    main()
