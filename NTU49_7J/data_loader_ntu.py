import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm


class NTUMotionDataset(Dataset):
    """
    NTU49_7J veri seti icin DataLoader.
    
    .npy shape: (N, C=3, T=64, V=7, M=1)
    Cikis shape: (C=3, V=7, T=64)  -> NAO pipeline ile uyumlu
    
    Joint mapping:
        0: ShoulderLeft
        1: ElbowLeft
        2: WristLeft
        3: Center (NTU-25 joint 21)
        4: ShoulderRight
        5: ElbowRight
        6: WristRight
    """

    def __init__(self, data_path, label_path):
        # Load data: (N, C=3, T=64, V=7, M=1)
        self.data = np.load(data_path).astype(np.float32)

        # Load labels: tuple of (names_list, labels_list)
        with open(label_path, 'rb') as f:
            self.names, self.labels = pickle.load(f)

        print(f"  Loaded {len(self.data)} samples | shape: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # (C=3, T=64, V=7, M=1) -> squeeze M -> (3, 64, 7) -> transpose to (3, 7, 64) = (C, V, T)
        x = self.data[idx]       # (3, 64, 7, 1)
        x = x.squeeze(-1)        # (3, 64, 7)
        x = np.transpose(x, (0, 2, 1))  # (3, 7, 64) = (C, V, T)

        y = self.labels[idx]
        name = self.names[idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), name


def get_ntu_loader(data_path, label_path, batch_size=32, shuffle=True):
    dataset = NTUMotionDataset(data_path, label_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
