import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm

class SequenceMotionDataset(Dataset):
    def __init__(self, h5_file, split='train', test_subject='K18'):
        self.h5_file = h5_file
        self.split = split
        self.test_subject = test_subject

        self.data, self.labels, self.names = self.load_sequences()

    def load_sequences(self):
        data, labels, names = [], [], []

        with h5py.File(self.h5_file, 'r') as f:
            all_keys = list(f.keys())
            if self.split == 'train':
                keys = [k for k in all_keys if not k.startswith(self.test_subject)]
            else:  # split == 'test'
                keys = [k for k in all_keys if k.startswith(self.test_subject)]

            for group_name in tqdm(keys, desc=f"⏳ Loading {self.split} data", total=len(keys)):
                group = f[group_name]
                frame_keys = sorted(group.keys(), key=lambda x: int(x))
                T = len(frame_keys)

                sequence = np.zeros((3, 7, T), dtype=np.float32)

                for t, frame_num in enumerate(frame_keys):
                    for j, joint_name in enumerate([
                        'Center', 'ShoulderLeft', 'ElbowLeft', 'WristLeft',
                        'ShoulderRight', 'ElbowRight', 'WristRight'
                    ]):
                        x = group[frame_num][f'{joint_name}/X'][()]
                        y = group[frame_num][f'{joint_name}/Y'][()]
                        z = group[frame_num][f'{joint_name}/Z'][()]
                        sequence[0, j, t] = x
                        sequence[1, j, t] = y
                        sequence[2, j, t] = z

                data.append(sequence)
                action_str = group_name.split('_')[1]
                action_id = int(action_str[1:]) - 1
                labels.append(action_id)
                names.append(group_name)

        return data, labels, names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        name = self.names[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), name


def get_loader(h5_path, split='train', batch_size=32, test_subject='K18'):
    dataset = SequenceMotionDataset(h5_path, split=split, test_subject=test_subject)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))

