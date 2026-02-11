"""
NTU49_7J icin en iyi model secimi.

Her kayitli checkpoint icin validation seti uzerinde:
  - L1 reconstruction error
  - Temporal smoothness
hesaplar ve en iyi epoch'u Results/best_epoch.txt'e yazar.
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net_G import ActFormer_Generator
from gp_sampling import sample_gp
from data_loader_ntu import get_ntu_loader

# ============================================================
#  Configuration
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LATENT_DIM = 64
NUM_CLASSES = 49
SEQ_LEN = 64
JOINTS = 7
CHANNELS = 3

SPLIT = "xsub"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, SPLIT)
VAL_DATA = os.path.join(DATA_DIR, "val_data_joint.npy")
VAL_LABEL = os.path.join(DATA_DIR, "val_label.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "Results", "saved_models")
BEST_EPOCH_FILE = os.path.join(BASE_DIR, "Results", "best_epoch.txt")


def get_available_epochs():
    """Kayitli Generator checkpoint'larinin epoch numaralarini dondurur."""
    epochs = []
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory not found: {MODEL_DIR}")
        return epochs
    for f in os.listdir(MODEL_DIR):
        if f.startswith("netG_epoch") and f.endswith(".pt"):
            epoch_num = int(f.replace("netG_epoch", "").replace(".pt", ""))
            epochs.append(epoch_num)
    return sorted(epochs)


def evaluate_epoch(epoch, val_loader):
    """Belirli bir epoch'un checkpoint'ini yukler ve validation metrikleri hesaplar."""
    model_path = os.path.join(MODEL_DIR, f"netG_epoch{epoch}.pt")

    net_G = ActFormer_Generator(
        Z=LATENT_DIM, T=SEQ_LEN, C=CHANNELS, V=JOINTS,
        num_class=NUM_CLASSES,
        embed_dim_ratio=64, depth=12, num_heads=14
    ).to(DEVICE)

    try:
        net_G.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"  Error loading epoch {epoch}: {e}")
        return None

    net_G.eval()

    total_l1 = 0.0
    total_smooth = 0.0
    total_samples = 0

    with torch.no_grad():
        for real_seq, labels, _ in val_loader:
            real_seq = real_seq.to(DEVICE)
            labels = labels.to(DEVICE)
            B = real_seq.size(0)

            z = sample_gp(B, SEQ_LEN, LATENT_DIM, DEVICE)
            fake_seq = net_G(z, labels)

            # L1 error
            l1 = torch.mean(torch.abs(real_seq - fake_seq)).item()

            # Temporal smoothness (velocity variance)
            vel = fake_seq[..., 1:] - fake_seq[..., :-1]
            smooth = torch.mean(vel ** 2).item()

            total_l1 += l1 * B
            total_smooth += smooth * B
            total_samples += B

    avg_l1 = total_l1 / total_samples
    avg_smooth = total_smooth / total_samples
    # Combined score (lower is better)
    combined = avg_l1 + 0.1 * avg_smooth

    return {"l1": avg_l1, "smooth": avg_smooth, "combined": combined}


def main():
    print(f"Device: {DEVICE}")
    print(f"Split: {SPLIT}")
    print(f"Model dir: {MODEL_DIR}\n")

    epochs = get_available_epochs()
    if not epochs:
        print("No checkpoints found!")
        return

    print(f"Found {len(epochs)} checkpoints: {epochs[0]} ~ {epochs[-1]}\n")

    val_loader = get_ntu_loader(VAL_DATA, VAL_LABEL, batch_size=BATCH_SIZE, shuffle=False)

    results = []
    best_score = float('inf')
    best_epoch = epochs[-1]

    for epoch in tqdm(epochs, desc="Evaluating checkpoints"):
        metrics = evaluate_epoch(epoch, val_loader)
        if metrics is None:
            continue

        results.append((epoch, metrics))
        print(f"  Epoch {epoch:5d} | L1: {metrics['l1']:.4f} | Smooth: {metrics['smooth']:.6f} | Combined: {metrics['combined']:.4f}")

        if metrics["combined"] < best_score:
            best_score = metrics["combined"]
            best_epoch = epoch

    # Save best epoch
    os.makedirs(os.path.dirname(BEST_EPOCH_FILE), exist_ok=True)
    with open(BEST_EPOCH_FILE, 'w') as f:
        f.write(str(best_epoch))

    print(f"\n{'='*50}")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Combined Score: {best_score:.4f}")
    print(f"  Saved to: {BEST_EPOCH_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
