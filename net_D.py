import torch
import torch.nn as nn
import h5py
import numpy as np
from graph_conv import GraphConv

class GCN_Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_classes=15, adj_path='Dataset/adjacency_matrix.h5'):
        super(GCN_Discriminator, self).__init__()

        # === Adjacency Matrix ===
        with h5py.File(adj_path, 'r') as f:
            adj = f['adjacency_matrix'][:]  # (7, 7)
            adj = np.expand_dims(adj, axis=0)  # (1, 7, 7)
        self.A = torch.tensor(adj, dtype=torch.float32)

        # === ST-GCN Blokları (Spatial + Temporal) ===
        # Mantık: Önce Graph Conv (Uzam), sonra Temporal Conv (Zaman)
        
        # Block 1
        self.gcn1 = GraphConv(extra_dim=1, in_channels=in_channels, out_channels=base_channels, 
                              kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), computation_kernel=self.A)
        self.tcn1 = nn.Conv2d(base_channels, base_channels, kernel_size=(1, 9), padding=(0, 4), stride=(1, 1))

        # Block 2 (Downsample Time)
        self.gcn2 = GraphConv(extra_dim=1, in_channels=base_channels, out_channels=base_channels*2, 
                              kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), computation_kernel=self.A)
        self.tcn2 = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=(1, 9), padding=(0, 4), stride=(1, 2))

        # Block 3 (Downsample Time)
        self.gcn3 = GraphConv(extra_dim=1, in_channels=base_channels*2, out_channels=base_channels*4, 
                              kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), computation_kernel=self.A)
        self.tcn3 = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=(1, 9), padding=(0, 4), stride=(1, 2))

        # Block 4
        self.gcn4 = GraphConv(extra_dim=1, in_channels=base_channels*4, out_channels=base_channels*8, 
                              kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), computation_kernel=self.A)
        self.tcn4 = nn.Conv2d(base_channels*8, base_channels*8, kernel_size=(1, 9), padding=(0, 4), stride=(1, 1))


        # === Nonlinearity & Regularization ===
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        # === Global Pooling ===
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # === Fully Connected Output ===
        self.fc = nn.Linear(base_channels * 8, 1)
        self.fc = nn.utils.spectral_norm(self.fc)

        # === Class Conditioning (Projection Discriminator) ===
        self.embedding = nn.Embedding(num_classes, base_channels * 8)
        nn.init.orthogonal_(self.embedding.weight)

    def forward(self, x, y):
        """
        x: (B, 3, 7, T)
        y: (B,)
        """
        # Block 1
        x = self.relu(self.gcn1(x))
        x = self.dropout(self.tcn1(x))
        
        # Block 2
        x = self.relu(self.gcn2(x))
        x = self.dropout(self.tcn2(x))

        # Block 3
        x = self.relu(self.gcn3(x))
        x = self.dropout(self.tcn3(x))

        # Block 4
        x = self.relu(self.gcn4(x))
        x = self.dropout(self.tcn4(x))

        # Global pooling (C,V,T -> C)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, C)

        # Projection discriminator
        proj = torch.sum(self.embedding(y) * x, dim=1, keepdim=True)  # (B, 1)
        score = self.fc(x) + proj

        return score.squeeze(1)
