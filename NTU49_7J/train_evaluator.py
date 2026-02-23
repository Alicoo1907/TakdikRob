"""
NTU49_7J icin FID Encoder egitimi.

Mantik:
  1. ActFormerEncoder7 (feature extractor) + classification head
  2. Train verisi ile action siniflandirma egitimi
  3. En iyi encoder agirliklarini kaydet -> fid.py kullanir
"""
import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from net_G import ActFormer_Generator
# === Parent dizinden importlar ===
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fid import ActFormerEncoder7
from data_loader_ntu import get_ntu_loader

# ============================================================
#  Hyperparameters
# ============================================================
EPOCHS = 150
BATCH_SIZE = 64
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 49
T = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT = "xsub"
DATA_DIR = os.path.join(BASE_DIR, SPLIT)
TRAIN_DATA = os.path.join(DATA_DIR, "train_data_joint.npy")
TRAIN_LABEL = os.path.join(DATA_DIR, "train_label.pkl")
VAL_DATA = os.path.join(DATA_DIR, "val_data_joint.npy")
VAL_LABEL = os.path.join(DATA_DIR, "val_label.pkl")

RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SAVE_PATH = os.path.join(RESULTS_DIR, "fid_encoder.pt")


# ============================================================
#  Model: Encoder + Classification Head
# ============================================================
class ActionClassifier(nn.Module):
    def __init__(self, num_classes=49):
        super().__init__()
        self.encoder = ActFormerEncoder7(T=T, V=7, C=3, out_dim=256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        feat = self.encoder(x)  # (B, 256)
        logits = self.fc(feat)  # (B, num_classes)
        return logits


# ============================================================
#  Training
# ============================================================
def train():
    print(f"Device: {DEVICE}")
    print(f"Split: {SPLIT} | Classes: {NUM_CLASSES} | T: {T}")

    # 1. Load Data
    train_loader = get_ntu_loader(TRAIN_DATA, TRAIN_LABEL, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_ntu_loader(VAL_DATA, VAL_LABEL, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Init Model
    model = ActionClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    losses = []

    # 3. Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for seq, labels, _ in pbar:
            seq, labels = seq.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({'loss': f'{total_loss/len(train_loader):.4f}', 'acc': f'{correct/total:.4f}'})

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        # 4. Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for seq, labels, _ in val_loader:
                seq, labels = seq.to(DEVICE), labels.to(DEVICE)
                logits = model(seq)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch} -> Train Acc: {correct/total:.4f} | Val Acc: {val_acc:.4f}")

        # Save best encoder
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.encoder.state_dict(), SAVE_PATH)
            print(f"  -> New best! Val Acc: {val_acc:.4f} | Saved encoder to {SAVE_PATH}")

    print(f"\nTraining finished. Best Val Acc: {best_acc:.4f}")

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("FID Encoder - Classifier Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "classifier_loss.png"))
    plt.close()
    print(f"Loss plot saved to {RESULTS_DIR}/classifier_loss.png")


if __name__ == "__main__":
    train()
