"""
NAO (7 Joint) - FID_w (Action-Wise FID) Hesabi

Her aksiyon sinifi icin ayri FID hesaplayip ortalamasini alir.
FID_w = (1/K) * sum(FID_k)  where K = number of classes

NAO datasi HDF5 formatinda, label group_name'den cikarilir:
  Ornek: K01_A01_walking_1 -> action_id = 0  (A01 -> int("01") - 1)
"""
import os
import torch
import numpy as np
import h5py
from tqdm import tqdm
from scipy import linalg

from fid import ActFormerEncoder7, frechet_distance, load_evaluator

# ============================================================
#  Configuration
# ============================================================
NUM_CLASSES = 15
MIN_SAMPLES_PER_CLASS = 2
T_MAX = 60
V = 7
C = 3


def load_real_and_fake_by_class(h5_path, fake_root):
    """
    HDF5'ten real, fake_root'tan fake verileri yukler.
    Label'i group_name'den cikarir (A01 -> 0, A02 -> 1, ...).
    
    Returns: dict[class_id] -> {"real": [tensors], "fake": [tensors]}
    """
    class_data = {c: {"real": [], "fake": []} for c in range(NUM_CLASSES)}

    with h5py.File(h5_path, 'r') as f:
        all_keys = list(f.keys())

        for group_name in tqdm(all_keys, desc="Loading real+fake pairs by class"):
            # Fake dosyasini bul
            fake_path = None
            for file in os.listdir(fake_root):
                cleaned = file.replace("(", "").replace(")", "").replace(",", "").replace("'", "")
                if group_name in cleaned:
                    fake_path = os.path.join(fake_root, file)
                    break

            if fake_path is None:
                continue

            # Label'i group_name'den cikar: K01_A01_walking_1 -> A01 -> 0
            action_str = group_name.split('_')[1]  # "A01"
            action_id = int(action_str[1:]) - 1      # 0

            # Real veriyi oku
            group = f[group_name]
            frame_keys = sorted(group.keys(), key=lambda x: int(x))
            T = min(len(frame_keys), T_MAX)
            real_seq = np.zeros((3, 7, T), dtype=np.float32)
            for t, frame_num in enumerate(frame_keys[:T]):
                for j, joint_name in enumerate([
                    'Center', 'ShoulderLeft', 'ElbowLeft', 'WristLeft',
                    'ShoulderRight', 'ElbowRight', 'WristRight'
                ]):
                    real_seq[0, j, t] = group[frame_num][f'{joint_name}/X'][()]
                    real_seq[1, j, t] = group[frame_num][f'{joint_name}/Y'][()]
                    real_seq[2, j, t] = group[frame_num][f'{joint_name}/Z'][()]

            # Fake veriyi oku
            fake_seq = np.load(fake_path)
            if fake_seq.shape[-1] != T:
                min_len = min(fake_seq.shape[-1], T)
                fake_seq = fake_seq[..., :min_len]
                real_seq = real_seq[..., :min_len]

            # Shape kontrol
            real_t = torch.tensor(real_seq, dtype=torch.float32)
            fake_t = torch.tensor(fake_seq, dtype=torch.float32)
            if real_t.shape[0] == 7 and real_t.shape[1] == 3:
                real_t = real_t.permute(1, 0, 2)
            if fake_t.shape[0] == 7 and fake_t.shape[1] == 3:
                fake_t = fake_t.permute(1, 0, 2)

            class_data[action_id]["real"].append(real_t)
            class_data[action_id]["fake"].append(fake_t)

    total = sum(len(class_data[c]["real"]) for c in range(NUM_CLASSES))
    print(f"Total matched: {total} sequences")
    return class_data


def encode_sequences(tensors, encoder, device):
    feats = []
    with torch.no_grad():
        for seq in tensors:
            seq = seq.unsqueeze(0).to(device)
            feat = encoder(seq).cpu().numpy()
            feats.append(feat)
    return np.concatenate(feats, axis=0)


def compute_fid_w(class_data, encoder, device):
    fid_per_class = {}
    skipped = []

    for class_id in tqdm(range(NUM_CLASSES), desc="Computing FID per class"):
        real_list = class_data[class_id]["real"]
        fake_list = class_data[class_id]["fake"]

        if len(real_list) < MIN_SAMPLES_PER_CLASS or len(fake_list) < MIN_SAMPLES_PER_CLASS:
            skipped.append(class_id)
            continue

        real_feats = encode_sequences(real_list, encoder, device)
        fake_feats = encode_sequences(fake_list, encoder, device)

        mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
        mu2, sigma2 = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)

        fid = frechet_distance(mu1, sigma1, mu2, sigma2)
        fid_per_class[class_id] = fid

    if skipped:
        print(f"\nSkipped {len(skipped)} classes (< {MIN_SAMPLES_PER_CLASS} samples): {skipped}")

    return fid_per_class


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h5_path = "Dataset/HDF5_Dataset_60frame/motions_data60frame.h5"
    fake_root = "Results/Full_Train_Generation"
    encoder_path = "Results/fid_encoder.pt"

    print(f"=== NAO FID_w (Action-Wise FID) ===")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Device: {device}\n")

    if not os.path.exists(encoder_path):
        print(f"Error: Encoder not found at {encoder_path}")
        return

    encoder = load_evaluator(encoder_path)
    class_data = load_real_and_fake_by_class(h5_path, fake_root)

    # Sinif basina sample sayilarini goster
    print("\nSamples per class:")
    for c in range(NUM_CLASSES):
        n = len(class_data[c]["real"])
        if n > 0:
            print(f"  Class {c:2d}: {n} pairs")

    fid_per_class = compute_fid_w(class_data, encoder, device)

    if fid_per_class:
        print(f"\n{'='*50}")
        print(f"{'Class':>8}  {'FID':>12}  {'Samples':>8}")
        print(f"{'='*50}")
        for c in sorted(fid_per_class.keys()):
            n = len(class_data[c]["real"])
            print(f"  {c:>5d}  {fid_per_class[c]:>12.4f}  {n:>8d}")

        fid_w = np.mean(list(fid_per_class.values()))
        print(f"{'='*50}")
        print(f"\nFID_w (Action-Wise): {fid_w:.4f}")
        print(f"  Computed over {len(fid_per_class)}/{NUM_CLASSES} classes")
    else:
        print("No class had enough samples to compute FID!")


if __name__ == "__main__":
    main()
