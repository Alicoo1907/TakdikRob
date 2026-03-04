import os
import torch
from data_loader import get_loader
from net_G import ActFormer_Generator
from net_D import GCN_Discriminator
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from gp_sampling import sample_gp
import h5py
import numpy as np



# === Hyperparameters ===
Z_DIM = 64
NUM_CLASSES = 15
EPOCHS = 6000
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "Results/saved_models"
Graph_Path = "Results/Train_Loss_Graph"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(Graph_Path, exist_ok=True)



csv_path = os.path.join(Graph_Path, "loss_log.csv")
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "G_Loss", "D_Loss"])
# === Load Data ===
train_loader = get_loader("Dataset/HDF5_Dataset_60frame/motions_data60frame.h5", batch_size=BATCH_SIZE, split="train")

# === Init Models ===
net_G = ActFormer_Generator(
    Z=Z_DIM,
    T=60,
    C=3,
    V=7,
    spectral_norm=True,
    learnable_pos_embed=True,
    out_normalize=None,
    num_class=NUM_CLASSES,
    embed_dim_ratio=64,  # 32 -> 64
    depth=12,            # 8 -> 12
    num_heads=14         # 12 -> 14 (Must divide 448 = 64*7)
).to(DEVICE)
net_D = GCN_Discriminator(
    in_channels=3,
    base_channels=64,
    num_classes=NUM_CLASSES,
    adj_path="Dataset/adjacency_matrix.h5"
).to(DEVICE)
# === Optimizers ===
optimizer_G = torch.optim.Adam(net_G.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(net_D.parameters(), lr=1e-4, betas=(0.5, 0.999))

# === Schedulers ===
scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=EPOCHS, eta_min=1e-6)
scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=EPOCHS, eta_min=1e-6)

# === Loss (Hinge loss örneği) ===
def d_loss_fn(real_out, fake_out):
    return torch.mean(torch.relu(1.0 - real_out)) + torch.mean(torch.relu(1.0 + fake_out))

def g_loss_fn(fake_out):
    return -torch.mean(fake_out)

# TODO
def temporal_smoothness_loss(seq):
    """
    seq: (B, C=3, V=7, T)
    """
    vel = seq[..., 1:] - seq[..., :-1]
    acc = vel[..., 1:] - vel[..., :-1]
    loss_vel = torch.mean(vel ** 2)
    loss_acc = torch.mean(acc ** 2)
    return 0.7 * loss_vel + 0.3 * loss_acc

# TODO
BONE_PAIRS = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6)]

def calculate_reference_bone_lengths(h5_path, bone_pairs):
    """
    HDF5 dosyasındaki tüm gerçek hareketler için kemik uzunluklarının
    ortalama değerini hesaplar.
    """
    all_lengths = {pair: [] for pair in bone_pairs}
    joint_names = [
        'Center', 'ShoulderLeft', 'ElbowLeft', 'WristLeft',
        'ShoulderRight', 'ElbowRight', 'WristRight'
    ]

    with h5py.File(h5_path, 'r') as f:
        all_keys = list(f.keys())

        for group_name in tqdm(all_keys, desc="Calculating Ref Bone Lengths"):
            group = f[group_name]
            frame_keys = sorted(group.keys(), key=lambda x: int(x))
            T = min(len(frame_keys), 60) # T'yi 60 ile sınırla (model T=60'a ayarlı)
            
            # (3, 7, T) formatında hareket dizisi
            real_seq = np.zeros((3, 7, T), dtype=np.float32)

            for t, frame_num in enumerate(frame_keys[:T]):
                for j, joint_name in enumerate(joint_names):
                    # Koordinatları oku
                    real_seq[0, j, t] = group[frame_num][f'{joint_name}/X'][()]
                    real_seq[1, j, t] = group[frame_num][f'{joint_name}/Y'][()]
                    real_seq[2, j, t] = group[frame_num][f'{joint_name}/Z'][()]

            # Numpy array'i olarak kemik uzunluğunu hesapla
            for (i, j) in bone_pairs:
                # bone_vec: (3, T)
                bone_vec = real_seq[:, i, :] - real_seq[:, j, :]
                
                # bone_len: (T) - L2 norm
                bone_len = np.linalg.norm(bone_vec, axis=0) 
                
                # O hareket dizisindeki tüm uzunlukları topla
                all_lengths[(i, j)].extend(bone_len.tolist())

    # Tüm veriler üzerinden her kemiğin ortalama uzunluğunu hesapla
    ref_lengths = {}
    for pair, lengths in all_lengths.items():
        if lengths:
            # Ortalama uzunluğu al ve torch tensor'a çevir
            mean_length = np.mean(lengths)
            ref_lengths[pair] = torch.tensor(mean_length, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        
    return ref_lengths

ref_lengths_dict = calculate_reference_bone_lengths("Dataset/HDF5_Dataset_60frame/motions_data60frame.h5", BONE_PAIRS)
print("Reference bone lengths calculated and ready.")

def bone_length_loss(seq, ref_lengths=None):
    """
    seq: (B, C=3, V=7, T)
    ref_lengths: her kemik için referans uzunluk listesi (isteğe bağlı)
    """
    loss = 0.0
    for (i, j) in BONE_PAIRS:
        bone_vec = seq[:, :, i, :] - seq[:, :, j, :]
        bone_len = torch.norm(bone_vec, dim=1)  # (B, T)
        if ref_lengths is not None:
            loss += torch.mean((bone_len - ref_lengths[(i, j)]) ** 2)
        else:
            loss += torch.var(bone_len)  # uzunluk sabit kalsın
    return loss / len(BONE_PAIRS)
# TODO
def center_joint_loss(motion, center_idx=0, w_static=1.0, w_smooth=0.3):
    """
    motion: Tensor (B, C=3, V=7, T)
    center_idx: Center joint'in indeksi (default 0)
    
    Center ekleminin:
      - Orijine (0,0,0) yakın olmasını (statik sabitleme)
      - Frame’ler arası ani sıçrama yapmamasını (smoothness)
    sağlayan kayıp.
    """
    # (B, 3, T) → center joint koordinatları
    center = motion[:, :, center_idx, :]

    # 1Konum sabitleme (Center orijine yakın olsun)
    static_loss = torch.mean(center ** 2)

    # 2Frame’ler arası hız farkı (Center hareketi pürüzsüz olsun)
    velocity = center[..., 1:] - center[..., :-1]
    smooth_loss = torch.mean(velocity ** 2)

    # Toplam ağırlıklı kayıp
    return w_static * static_loss + w_smooth * smooth_loss

g_losses = []
d_losses = []
# === Training loop ===
for epoch in tqdm(range(1, EPOCHS + 1), desc="Epoch Progress"):
    batch_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for real_seq, labels, _ in batch_bar:
        real_seq = real_seq.to(DEVICE)  # (B, 3, 7, T)
        labels = labels.to(DEVICE)
        B, C, V, T = real_seq.shape

        # === Train Discriminator ===
        z = sample_gp(B, T, Z_DIM, DEVICE)
        fake_seq = net_G(z, labels)

        # R1 Regularization for Discriminator stability
        real_seq.requires_grad = True
        real_out = net_D(real_seq, labels)
        
        # Calculate R1 Gradient Penalty
        grad_real = torch.autograd.grad(
            outputs=real_out.sum(), inputs=real_seq,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        r1_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()

        fake_out = net_D(fake_seq.detach(), labels)
        d_loss = d_loss_fn(real_out, fake_out) + 10.0 * r1_penalty # R1_gamma = 10.0

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # === Train Generator ===
        fake_out = net_D(fake_seq, labels)
        g_adv_loss = g_loss_fn(fake_out)

        g_l1_loss = torch.mean(torch.abs(real_seq - fake_seq))

        loss_center = center_joint_loss(fake_seq, center_idx=0)
        loss_bone = bone_length_loss(fake_seq, ref_lengths=ref_lengths_dict)
        loss_temp = temporal_smoothness_loss(fake_seq)
        
        total_g_loss = (
            1.0 * g_adv_loss             # Adversarial goal
            + 10.0 * g_l1_loss           # Coordinate precision
            + 0.1 * loss_center          
            + 0.2 * loss_bone
            + 0.2 * loss_temp
        )
        optimizer_G.zero_grad()
        total_g_loss.backward()
        # Gradient Clipping (Generator) to prevent exploding gradients
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
    if epoch % 1 == 0 or epoch == EPOCHS:
        torch.save(net_G.state_dict(), os.path.join(SAVE_PATH, f"netG_epoch{epoch}.pt"))
        torch.save(net_D.state_dict(), os.path.join(SAVE_PATH, f"netD_epoch{epoch}.pt"))
        print(f"Models saved at epoch {epoch}")
        # Loss grafiğini çiz
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 1), g_losses, label="G Loss", color='orange')
        plt.plot(range(1, epoch + 1), d_losses, label="D Loss", color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Generator & Discriminator Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Graph_Path, f"loss_epoch{epoch}.png"))
        plt.close()
