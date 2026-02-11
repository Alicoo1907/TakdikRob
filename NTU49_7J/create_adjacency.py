"""
NTU49_7J icin adjacency matrix olusturur ve adjacency_matrix.h5 olarak kaydeder.

Joint mapping:
    0: ShoulderLeft
    1: ElbowLeft
    2: WristLeft
    3: Center
    4: ShoulderRight
    5: ElbowRight
    6: WristRight

Kemik baglantilari:
    0-1: ShoulderLeft  <-> ElbowLeft
    1-2: ElbowLeft     <-> WristLeft
    0-3: ShoulderLeft  <-> Center
    3-4: Center        <-> ShoulderRight
    4-5: ShoulderRight <-> ElbowRight
    5-6: ElbowRight    <-> WristRight
"""
import numpy as np
import h5py
import os

EDGES = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (5, 6)]

adj = np.zeros((7, 7), dtype=np.float32)
for i, j in EDGES:
    adj[i, j] = 1.0
    adj[j, i] = 1.0

print("NTU49_7J Adjacency Matrix:")
print(adj)

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adjacency_matrix.h5")
with h5py.File(save_path, 'w') as f:
    f.create_dataset('adjacency_matrix', data=adj)

print(f"\nSaved to: {save_path}")
