# networks.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def get_grid_3d(shape, device):
    """
    Generates a 3D grid for a batch.
    shape: (batchsize, size_x, size_y, size_z)
    """
    batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    gridx, gridy, gridz = torch.meshgrid(gridx, gridy, gridz, indexing='ij')
    grid = torch.stack((gridx, gridy, gridz), dim=-1).to(device)
    # Repeat for batch and return shape (batch, size_x, size_y, size_z, 3)
    return grid.unsqueeze(0).repeat([batchsize, 1, 1, 1, 1])


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d_onestep(nn.Module):
    """
    A simple, one-step FNO model that predicts the next u(t+dt) from u(t).
    Designed for the non-dimensionalized Allen-Cahn equation.
    """

    def __init__(self, modes1, modes2, modes3, width, n_layers):
        super(FNO3d_onestep, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.n_layers = n_layers

        self.p = nn.Linear(4, self.width)  # Input: u, x, y, z

        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(self.n_layers):
            self.convs.append(SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3))
            self.ws.append(nn.Conv3d(self.width, self.width, 1))

        self.q = nn.Linear(self.width, 1)  # Output: just the next u

    def forward(self, x):
        grid = get_grid_3d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.q(x)  # Shape: (batch, S, S, S, 1)
        return x