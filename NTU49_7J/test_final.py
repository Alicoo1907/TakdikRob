import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# === Parent dizinden model importlari ===
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net_G import ActFormer_Generator
from data_loader_ntu import get_ntu_loader


# ============================================================
#  Configuration
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LATENT_DIM = 64
NUM_CLASSES = 49
SEQ_LEN = 64  # T=64
JOINTS = 7
CHANNELS = 3

SPLIT = "xsub"  # "xsub" veya "xview"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, SPLIT)
TRAIN_DATA = os.path.join(DATA_DIR, "train_data_joint.npy")
TRAIN_LABEL = os.path.join(DATA_DIR, "train_label.pkl")


def get_best_epoch():
    try:
        with open(os.path.join(BASE_DIR, "Results", "best_epoch.txt"), "r") as f:
            return int(f.read().strip())
    except:
        return 3000


BEST_EPOCH = get_best_epoch()
print(f"🔥 Evaluation for Epoch: {BEST_EPOCH}")

MODEL_PATH = os.path.join(BASE_DIR, "Results", "saved_models", f"netG_epoch{BEST_EPOCH}.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "Results", "Full_Train_Generation")


def generate_full_test_set():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Temiz baslangic icin klasoru bosalt
    import shutil
    for filename in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    print(f"Loading Generator from: {MODEL_PATH}")

    # 1. Load Generator
    net_G = ActFormer_Generator(
        Z=LATENT_DIM, T=SEQ_LEN, C=CHANNELS, V=JOINTS,
        num_class=NUM_CLASSES,
        embed_dim_ratio=64,
        depth=12,
        num_heads=14
    ).to(DEVICE)

    try:
        net_G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    net_G.eval()

    # 2. Load train data (ayni label'larla generation)
    loader = get_ntu_loader(TRAIN_DATA, TRAIN_LABEL, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Loader size: {len(loader)} batches")

    generated_count = 0

    # 3. Generation Loop
    print("Starting generation...")
    with torch.no_grad():
        for batch_idx, (real_motion, labels, group_names) in enumerate(tqdm(loader)):
            batch_size = real_motion.size(0)

            noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            labels = labels.to(DEVICE)

            fake_motion = net_G(noise, labels)  # (B, C, V, T)

            fake_motion_np = fake_motion.cpu().numpy()

            for i in range(batch_size):
                sample = fake_motion_np[i]  # (3, 7, 64)
                gn = group_names[i]
                file_name = f"{gn}_label_{labels[i].item()}.npy"
                np.save(os.path.join(OUTPUT_DIR, file_name), sample)
                generated_count += 1

    print(f"\nFinished! Generated {generated_count} motion sequences in '{OUTPUT_DIR}'.")
    print("You can now run FID calculation pointing to this directory.")


if __name__ == "__main__":
    generate_full_test_set()
