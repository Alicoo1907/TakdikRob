import os
import torch
import numpy as np
import h5py
from tqdm import tqdm
from scipy import linalg
from fid import ActFormerEncoder7, load_evaluator, compute_motion_fid, compute_motion_fid_per_sequence

import pandas as pd

def load_data_k18(h5_path, fake_root, subject_id="K18"):
    data_pairs = []
    
    with h5py.File(h5_path, 'r') as f:
        all_keys = [k for k in f.keys() if k.startswith(subject_id)]
        print(f"Loading {len(all_keys)} sequences for subject {subject_id}...")

        for group_name in tqdm(all_keys, desc="Loading K18"):
            fake_path = None
            fake_filename = None
            for file in os.listdir(fake_root):
                if group_name in file:
                    fake_path = os.path.join(fake_root, file)
                    fake_filename = file
                    break
            
            if fake_path is None:
                continue

            group = f[group_name]
            frame_keys = sorted(group.keys(), key=lambda x: int(x))
            T = min(len(frame_keys), 60)
            real_seq = np.zeros((3, 7, T), dtype=np.float32)
            for t, frame_num in enumerate(frame_keys[:T]):
                for j, joint_name in enumerate([
                    'Center', 'ShoulderLeft', 'ElbowLeft', 'WristLeft',
                    'ShoulderRight', 'ElbowRight', 'WristRight'
                ]):
                    real_seq[0, j, t] = group[frame_num][f'{joint_name}/X'][()]
                    real_seq[1, j, t] = group[frame_num][f'{joint_name}/Y'][()]
                    real_seq[2, j, t] = group[frame_num][f'{joint_name}/Z'][()]

            fake_seq = np.load(fake_path)
            if fake_seq.shape[-1] != T:
                min_len = min(fake_seq.shape[-1], T)
                fake_seq = fake_seq[..., :min_len]
                real_seq = real_seq[..., :min_len]

            data_pairs.append({
                "sequence": group_name,
                "fake_file": fake_filename,
                "real": torch.tensor(real_seq),
                "fake": torch.tensor(fake_seq)
            })

    return data_pairs

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h5_path = "Dataset/HDF5_Dataset_60frame/motions_data60frame.h5"
    fake_root = "Results/K18_Generation"
    encoder_path = "Results/fid_encoder.pt"
    save_csv = "Results/k18_motion_fid_per_sequence.csv"

    if not os.path.exists(encoder_path):
        print(f"Error: {encoder_path} not found!")
        return

    encoder = load_evaluator(encoder_path)
    
    print(f"Device: {device}")
    print(f"Reading real data from {h5_path}")
    print(f"Reading fake data from {fake_root}")
    
    pairs = load_data_k18(h5_path, fake_root)
    
    if not pairs:
        print("No K18 sequences found!")
        return
    
    print(f"Loaded {len(pairs)} sequences.")

    real_tensors = [p["real"] for p in pairs]
    fake_tensors = [p["fake"] for p in pairs]

    if len(real_tensors) > 0:
        global_fid = compute_motion_fid(real_tensors, fake_tensors, device=device, encoder=encoder)
        print(f"\nGlobal Motion FID Score (Transformer feat): {global_fid:.4f}")

    print("\nCalculating Motion FID per sequence (Transformer feat)...")
    results = []
    for p in tqdm(pairs, desc="Processing per-sequence FID"):
        score = compute_motion_fid([p["real"]], [p["fake"]], device=device, encoder=encoder, verbose=False)
        results.append({
            "sequence": p["sequence"],
            "fake_file": p["fake_file"],
            "FID": float(score)
        })

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(save_csv, index=False)
    
    print(f"\nSaved per-sequence FID results: {save_csv}")
    if len(df) > 0:
        print(f"Average Motion FID (across {len(df)} sequences): {df['FID'].mean():.4f}")

if __name__ == "__main__":
    main()
