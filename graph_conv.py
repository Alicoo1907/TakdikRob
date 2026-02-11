import torch
import torch.nn as nn
# TODO
class GraphConv(nn.Module):
    def __init__(self,
                 extra_dim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 computation_kernel=None,
                 use_graph_conv=True,
                 spectral_norm=False):
        super(GraphConv, self).__init__()

        assert extra_dim == 1, "Only extra_dim=1 (temporal conv) supported in this simplified version"
        assert computation_kernel is not None, "computation_kernel (adjacency) must be provided"

        self.extra_dim = extra_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.computation_kernel = computation_kernel  # shape: (K, W, V)
        self.use_graph_conv = use_graph_conv
        self.K = computation_kernel.shape[0]
        self.W = computation_kernel.shape[1]
        self.V = computation_kernel.shape[2]

        A = torch.tensor(self.computation_kernel, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * self.K,  # K * out_channels
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        # x shape: (N, C_in, V, T)
        N, _, V, T = x.shape

        x = self.conv(x)  # -> (N, C_out*K, V, T_out)
        N, CK, V, T_out = x.shape

        expected_CK = self.out_channels * self.K
        if CK != expected_CK:
            print(f"[GraphConv] Shape mismatch in view! Got CK={CK}, expected {expected_CK}")
            print(f"→ x.shape before view: {x.shape}")
            raise RuntimeError(f"Invalid shape for view: CK={CK}, expected {expected_CK}")

        # Şekillendir ve graph matrisle çarp
        x = x.view(N, self.out_channels, self.K, V, T_out)  # (N, C_out, K, V, T_out)
        x = torch.einsum('kwv,nckvt->ncwt', self.A, x)       # (N, C_out, W, T_out)

        return x
