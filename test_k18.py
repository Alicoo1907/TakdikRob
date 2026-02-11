import torch
import torch.nn as nn
from net_G import ActFormer_Generator
from data_loader import get_loader
import numpy as np
import os
from tqdm import tqdm

def get_best_epoch():
    try:
        with open("Results/best_epoch.txt", "r") as f:
            return int(f.read().strip())
    except:
        return 1915

BEST_EPOCH = get_best_epoch()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LATENT_DIM = 64
NUM_CLASSES = 15
SEQ_LEN = 60
JOINTS = 7
CHANNELS = 3

MODEL_PATH = f"Results/saved_models/netG_epoch{BEST_EPOCH}.pt"
OUTPUT_DIR = "Results/K18_Generation"
DATA_PATH = "Dataset/HDF5_Dataset_60frame/motions_data60frame.h5"

def generate_k18_set():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Generator (Extreme Architecture)
    net_G = ActFormer_Generator(
        Z=LATENT_DIM, T=SEQ_LEN, C=CHANNELS, V=JOINTS,
        num_class=NUM_CLASSES,
        embed_dim_ratio=64,
        depth=12,
        num_heads=14
    ).to(DEVICE)
    
    net_G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net_G.eval()
    print(f"Model loaded: {MODEL_PATH}")

    # 2. Get Test Loader (ONLY K18)
    test_loader = get_loader(DATA_PATH, split="test", test_subject="K18", batch_size=BATCH_SIZE)
    print(f"K18 Loader size: {len(test_loader)} batches")

    generated_count = 0
    with torch.no_grad():
        for batch_idx, (real_motion, labels, group_names) in enumerate(tqdm(test_loader)):
            batch_size = real_motion.size(0)
            noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            labels = labels.to(DEVICE)
            fake_motion = net_G(noise, labels)
            fake_motion_np = fake_motion.cpu().numpy()
            
            for i in range(batch_size):
                sample = fake_motion_np[i]
                gn = group_names[i]
                file_name = f"{gn}_label_{labels[i].item()}.npy"
                np.save(os.path.join(OUTPUT_DIR, file_name), sample)
                generated_count += 1
                
    print(f"Generated {generated_count} K18 sequences in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    generate_k18_set()
