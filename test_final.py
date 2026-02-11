import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from net_G import ActFormer_Generator
from data_loader import get_loader
import numpy as np
import os
from tqdm import tqdm
import time
# from transformer_utils import get_masks_and_count

# === CONFIGURATION ===
def get_best_epoch():
    try:
        with open("Results/best_epoch.txt", "r") as f:
            return int(f.read().strip())
    except:
        return 3000  # En son epoch varsayalım

BEST_EPOCH = get_best_epoch()
print(f" Evaluation for Epoch: {BEST_EPOCH}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LATENT_DIM = 64  # Must match training
NUM_CLASSES = 15
SEQ_LEN = 60
JOINTS = 7
CHANNELS = 3

# Paths
MODEL_PATH = f"Results/saved_models/netG_epoch{BEST_EPOCH}.pt"
OUTPUT_DIR = "Results/Full_Train_Generation"
DATA_PATH = "Dataset/HDF5_Dataset_60frame/motions_data60frame.h5"

def generate_full_test_set():
    # Klasörün üst dizinlerini de (Results/) oluşturduğundan emin ol
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory confirmed: {OUTPUT_DIR}")
    
    # İçini boşalt (temiz başlangıç için)
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
        embed_dim_ratio=64,  # Match main.py Phase 2
        depth=12,            # Match main.py Phase 2
        num_heads=14         # Match main.py Phase 2
    ).to(DEVICE)
    
    try:
        net_G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    net_G.eval()

    # 2. Get Test Loader (Full Test Set)
    # Note: We use the test split, but we don't filter by subject 'K18'.
    # We want ALL test data to have a large sample size (~1000+)
    test_loader = get_loader(DATA_PATH, split="train", batch_size=BATCH_SIZE)
    print(f"Test Loader size: {len(test_loader)} batches")

    generated_count = 0
    
    # 3. Generation Loop
    print("Starting generation...")
    with torch.no_grad():
        for batch_idx, (real_motion, labels, group_names) in enumerate(tqdm(test_loader)):
            # real_motion: (B, C, V, T) -> (B, 3, 7, 60)
            batch_size = real_motion.size(0)
            
            # Create noise and move to device
            noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            labels = labels.to(DEVICE)
            
            # Generate
            fake_motion = net_G(noise, labels)  # (B, C, V, T)
            
            # Save each sample as .npy
            fake_motion_np = fake_motion.cpu().numpy() # (B, 3, 7, 60)
            
            for i in range(batch_size):
                sample = fake_motion_np[i] # (3, 7, 60)
                gn = group_names[i]
                # Format: K01_walking_1_1_label_0.npy
                file_name = f"{gn}_label_{labels[i].item()}.npy"
                np.save(os.path.join(OUTPUT_DIR, file_name), sample)
                generated_count += 1
                
    print(f"Finished! Generated {generated_count} motion sequences in '{OUTPUT_DIR}'.")
    print("You can now run 'fid.py' pointing to this directory.")

if __name__ == "__main__":
    generate_full_test_set()
