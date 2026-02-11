import torch
import torch.nn as nn
from data_loader import get_loader
from fid import ActFormerEncoder7
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# === Hyperparameters ===
EPOCHS = 150
BATCH_SIZE = 64
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "Results/fid_encoder.pt"
NUM_CLASSES = 15

# === Model Definition ===
class ActionClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        # Use the same encoder architecture as in fid.py
        self.encoder = ActFormerEncoder7(T=60, V=7, C=3, out_dim=256)
        # Add a classification head
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, 3, 7, T)
        feat = self.encoder(x)  # (B, 256)
        logits = self.fc(feat)  # (B, num_classes)
        return logits

def train():
    print(f"Device: {DEVICE}")
    
    # 1. Load Data
    train_loader = get_loader("Dataset/HDF5_Dataset_60frame/motions_data60frame.h5", 
                              split="train", batch_size=BATCH_SIZE)
    # We use 'test' split for validation to check accuracy
    val_loader = get_loader("Dataset/HDF5_Dataset_60frame/motions_data60frame.h5", 
                            split="test", batch_size=BATCH_SIZE, test_subject='K18')

    # 2. Init Model
    model = ActionClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, 
                                                    steps_per_epoch=len(train_loader), 
                                                    epochs=EPOCHS)
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
            
            # Accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': total_loss/len(train_loader), 'acc': correct/total})
            
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

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            # We save only the encoder part for FID
            torch.save(model.encoder.state_dict(), SAVE_PATH)
            print(f"Saved new best encoder to {SAVE_PATH}")

    print("Training finished.")
    
    # Plot loss
    plt.plot(losses)
    plt.title("Classifier Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("Results/classifier_loss.png")

if __name__ == "__main__":
    train()
