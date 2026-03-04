import os
import sys
import torch
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import numpy as np

from net_G import ActFormer_Generator
# === Parent dizinden model importlari ===
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net_D import GCN_Discriminator
from gp_sampling import sample_gp
from data_loader_ntu import get_ntu_loader


# ============================================================
#  Hyperparameters
# ============================================================
Z_DIM = 64
NUM_CLASSES = 49
EPOCHS = 2000
BATCH_SIZE = 32
T = 64  # NTU frame sayisi (NAO=60)
V = 7
C = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME_FROM = 0  # Devam etmek istediginiz epoch numarasi (0 = sifirdan basla)
START_EPOCH = RESUME_FROM + 1

# === Paths ===
SPLIT = "xsub"  # "xsub" veya "xview"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), SPLIT)
TRAIN_DATA = os.path.join(DATA_DIR, "train_data_joint.npy")
TRAIN_LABEL = os.path.join(DATA_DIR, "train_label.pkl")
ADJ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adjacency_matrix.h5")

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results", "saved_models")
Graph_Path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results", "Train_Loss_Graph")
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(Graph_Path, exist_ok=True)

csv_path = os.path.join(Graph_Path, "loss_log.csv")
if RESUME_FROM == 0:
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "G_Loss", "D_Loss"])


# ============================================================
#  Load Data
# ============================================================
print(f"Loading NTU49_7J data ({SPLIT})...")
train_loader = get_ntu_loader(TRAIN_DATA, TRAIN_LABEL, batch_size=BATCH_SIZE, shuffle=True)


# ============================================================
#  Init Models
# ============================================================
net_G = ActFormer_Generator(
    Z=Z_DIM,
    T=T,       # 64
    C=C,
    V=V,
    spectral_norm=True,
    learnable_pos_embed=True,
    out_normalize=None,
    num_class=NUM_CLASSES,  # 49
    embed_dim_ratio=64,
    depth=12,
    num_heads=14   # 448 / 14 = 32 (embed_dim = 64*7 = 448)
).to(DEVICE)

net_D = GCN_Discriminator(
    in_channels=C,
    base_channels=64,
    num_classes=NUM_CLASSES,  # 49
    adj_path=ADJ_PATH
).to(DEVICE)


# ============================================================
#  Optimizers & Schedulers
# ============================================================
optimizer_G = torch.optim.Adam(net_G.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(net_D.parameters(), lr=1e-4, betas=(0.5, 0.999))
scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=EPOCHS, eta_min=1e-6)
scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=EPOCHS, eta_min=1e-6)

# === Resume Logic ===
if RESUME_FROM > 0:
    print(f"Resuming from epoch {RESUME_FROM}...")
    net_G.load_state_dict(torch.load(os.path.join(SAVE_PATH, f"netG_epoch{RESUME_FROM}.pt"), map_location=DEVICE))
    net_D.load_state_dict(torch.load(os.path.join(SAVE_PATH, f"netD_epoch{RESUME_FROM}.pt"), map_location=DEVICE))
    
    # schedulerlari ilerletmek icin (her batch icin bir adim)
    # len(train_loader) * RESUME_FROM kadar step atmamiz gerekiyor
    steps_done = len(train_loader) * RESUME_FROM
    for _ in range(steps_done):
        scheduler_G.step()
        scheduler_D.step()
    print(f"Schedulers advanced by {steps_done} steps.")


# ============================================================
#  Loss Functions
# ============================================================
def d_loss_fn(real_out, fake_out):
    return torch.mean(torch.relu(1.0 - real_out)) + torch.mean(torch.relu(1.0 + fake_out))

def g_loss_fn(fake_out):
    return -torch.mean(fake_out)


# --- Temporal Smoothness Loss ---
def temporal_smoothness_loss(seq):
    """seq: (B, C=3, V=7, T)"""
    vel = seq[..., 1:] - seq[..., :-1]
    acc = vel[..., 1:] - vel[..., :-1]
    loss_vel = torch.mean(vel ** 2)
    loss_acc = torch.mean(acc ** 2)
    return 0.7 * loss_vel + 0.3 * loss_acc


# --- Bone Length Loss ---
# NTU Joint mapping:
#   0: ShoulderLeft, 1: ElbowLeft, 2: WristLeft
#   3: Center
#   4: ShoulderRight, 5: ElbowRight, 6: WristRight
BONE_PAIRS = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (5, 6)]


def calculate_reference_bone_lengths(data_path, bone_pairs):
    """
    .npy dosyasindaki tum hareketler icin kemik uzunluklarinin 
    ortalama degerini hesaplar.
    
    data shape: (N, C=3, T=64, V=7, M=1)
    """
    data = np.load(data_path)  # (N, 3, 64, 7, 1)
    data = data.squeeze(-1)     # (N, 3, 64, 7) -> (N, C, T, V)

    ref_lengths = {}
    for (i, j) in tqdm(bone_pairs, desc="Calculating Ref Bone Lengths"):
        # bone_vec: (N, 3, T)
        bone_vec = data[:, :, :, i] - data[:, :, :, j]
        # bone_len: (N, T)
        bone_len = np.linalg.norm(bone_vec, axis=1)
        mean_length = np.mean(bone_len)
        ref_lengths[(i, j)] = torch.tensor(mean_length, dtype=torch.float32, device=DEVICE)

    return ref_lengths


ref_lengths_dict = calculate_reference_bone_lengths(TRAIN_DATA, BONE_PAIRS)
print("Reference bone lengths calculated and ready.")
for pair, length in ref_lengths_dict.items():
    print(f"  Bone {pair}: {length.item():.4f}")


def bone_length_loss(seq, ref_lengths=None):
    """seq: (B, C=3, V=7, T)"""
    loss = 0.0
    for (i, j) in BONE_PAIRS:
        bone_vec = seq[:, :, i, :] - seq[:, :, j, :]
        bone_len = torch.norm(bone_vec, dim=1)  # (B, T)
        if ref_lengths is not None:
            loss += torch.mean((bone_len - ref_lengths[(i, j)]) ** 2)
        else:
            loss += torch.var(bone_len)
    return loss / len(BONE_PAIRS)


# --- Center Joint Loss ---
def center_joint_loss(motion, center_idx=3, w_static=1.0, w_smooth=0.3):
    """
    motion: (B, C=3, V=7, T)
    NTU'da center_idx=3
    """
    center = motion[:, :, center_idx, :]
    static_loss = torch.mean(center ** 2)
    velocity = center[..., 1:] - center[..., :-1]
    smooth_loss = torch.mean(velocity ** 2)
    return w_static * static_loss + w_smooth * smooth_loss


# ============================================================
#  Training Loop
# ============================================================
g_losses = []
d_losses = []

if RESUME_FROM > 0:
    if os.path.exists(csv_path):
        rows = []
        with open(csv_path, mode='r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if int(row[0]) <= RESUME_FROM:
                    rows.append(row)
        
        # CSV'yi resume edilen epoch'a kadar temizle (duplike olmamasi icin)
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
            
        # Listeleri doldur
        for row in rows:
            g_losses.append(float(row[1]))
            d_losses.append(float(row[2]))
            
        print(f"Loaded and cleaned {len(g_losses)} previous loss records from CSV.")

print(f"\n{'='*60}")
print(f"  NTU49_7J Training | Split: {SPLIT}")
print(f"  Classes: {NUM_CLASSES} | T: {T} | Epochs: {EPOCHS}")
print(f"  Device: {DEVICE}")
print(f"{'='*60}\n")

for epoch in tqdm(range(START_EPOCH, EPOCHS + 1), desc="Epoch Progress"):
    batch_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for real_seq, labels, _ in batch_bar:
        real_seq = real_seq.to(DEVICE)  # (B, 3, 7, T=64)
        labels = labels.to(DEVICE)
        B, C_dim, V_dim, T_dim = real_seq.shape

        # === Train Discriminator ===
        z = sample_gp(B, T_dim, Z_DIM, DEVICE)
        fake_seq = net_G(z, labels)

        # R1 Regularization
        real_seq.requires_grad = True
        real_out = net_D(real_seq, labels)

        fake_out = net_D(fake_seq.detach(), labels)
        d_loss = d_loss_fn(real_out, fake_out) 

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # === Train Generator ===
        fake_out = net_D(fake_seq, labels)
        g_adv_loss = g_loss_fn(fake_out)
        g_l1_loss = torch.mean(torch.abs(real_seq - fake_seq))

        loss_center = center_joint_loss(fake_seq, center_idx=3)
        loss_bone = bone_length_loss(fake_seq, ref_lengths=ref_lengths_dict)
        loss_temp = temporal_smoothness_loss(fake_seq)

        total_g_loss = (
            1.0 * g_adv_loss
            + 10.0 * g_l1_loss
            + 0.1 * loss_center
            + 0.2 * loss_bone
            + 0.2 * loss_temp
        )

        optimizer_G.zero_grad()
        total_g_loss.backward()
        torch.nn.utils.clip_grad_norm_(net_G.parameters(), max_norm=1.0)
        optimizer_G.step()

        # Step schedulers
        scheduler_G.step()
        scheduler_D.step()

    g_losses.append(total_g_loss.item())
    d_losses.append(d_loss.item())
    print(f"[Epoch {epoch}] D_loss: {d_loss.item():.4f} | G_loss: {total_g_loss.item():.4f}")

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, total_g_loss.item(), d_loss.item()])

    # === Save checkpoint ===
    if epoch % 20 == 0 or epoch == EPOCHS:
        torch.save(net_G.state_dict(), os.path.join(SAVE_PATH, f"netG_epoch{epoch}.pt"))
        torch.save(net_D.state_dict(), os.path.join(SAVE_PATH, f"netD_epoch{epoch}.pt"))
        print(f"Models saved at epoch {epoch}")

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 1), g_losses, label="G Loss", color='orange')
        plt.plot(range(1, epoch + 1), d_losses, label="D Loss", color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"NTU49_7J ({SPLIT}) - Generator & Discriminator Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Graph_Path, f"loss_epoch{epoch}.png"))
        plt.close()
