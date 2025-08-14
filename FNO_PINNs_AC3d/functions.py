# functions.py

# ... (get_grid3d_from_config, initial_condition_grid, laplacian_fourier_3d are correct and remain unchanged) ...
import torch
import torch.nn.functional as F
import numpy as np
import config


def get_grid3d_from_config(device):
    S = config.GRID_RESOLUTION
    gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridz = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridx, gridy, gridz = torch.meshgrid(gridx, gridy, gridz, indexing='ij')
    grid = torch.stack((gridx, gridy, gridz), dim=-1).to(device)
    return grid


def initial_condition_grid(grid):
    x, y, z = grid[..., 0], grid[..., 1], grid[..., 2]
    dist_phys = torch.sqrt(
        (x - config.DROP_CENTER_X) ** 2 +
        (y - config.DROP_CENTER_Y) ** 2 +
        (z - config.DROP_CENTER_Z) ** 2
    )
    ic = torch.tanh(
        (config.DROP_RADIUS_PARAM - dist_phys) / (2 * config.EPSILON_PARAM)
    )
    return ic.unsqueeze(-1)


def laplacian_fourier_3d(u, dx):
    nx, ny, nz = u.shape[1], u.shape[2], u.shape[3]
    k_x = torch.fft.fftfreq(nx, d=dx).to(u.device)
    k_y = torch.fft.fftfreq(ny, d=dx).to(u.device)
    k_z = torch.fft.fftfreq(nz, d=dx).to(u.device)
    kx, ky, kz = torch.meshgrid(k_x, k_y, k_z, indexing='ij')
    minus_k_squared = -(kx ** 2 + ky ** 2 + kz ** 2) * (2 * np.pi) ** 2  # Add (2*pi)^2 for standard Fourier derivative
    u_ft = torch.fft.fftn(u, dim=[1, 2, 3])
    u_lap_ft = minus_k_squared * u_ft
    u_lap = torch.fft.ifftn(u_lap_ft, dim=[1, 2, 3])
    return u_lap.real


# --- FINAL DEBUGGING LOSS FUNCTION ---
def fno_onestep_loss(u_in, u_pred, grid):
    S = u_in.shape[1]
    dx = 1.0 / S

    # 1. Time derivative
    u_t = (u_pred - u_in) / config.DT

    # 2. Spatial terms
    laplacian_u = laplacian_fourier_3d(u_pred.squeeze(-1), dx)
    reaction_term = u_pred.squeeze(-1) ** 3 - u_pred.squeeze(-1)

    mu_spatial = config.EPSILON_PARAM ** 2 * laplacian_u - reaction_term

    # 3. Residual
    residual = u_t.squeeze(-1) - config.LAMBDA_PARAM * mu_spatial

    # Balance the residual before taking the MSE
    balanced_residual = residual / config.LAMBDA_PARAM
    loss = F.mse_loss(balanced_residual, torch.zeros_like(balanced_residual))

    # --- DEBUGGING: Return the individual terms' magnitudes ---
    # We return their mean squared value to get a single number for each.
    debug_ut_mse = torch.mean((u_t.squeeze(-1) / config.LAMBDA_PARAM) ** 2)
    debug_muspatial_mse = 25*torch.mean(mu_spatial ** 2)

    return loss, debug_ut_mse, debug_muspatial_mse