# dataset.py

import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube
import config


def sample_domain(N, time_interval, requires_grad=False, use_lhs=True):
    """Samples N collocation points within the 3D spatial domain and a time interval."""
    dim = 4  # x, y, z, t
    lb = np.array([config.DOMAIN_BOUNDS[0], config.DOMAIN_BOUNDS[0], config.DOMAIN_BOUNDS[0], time_interval[0]])
    ub = np.array([config.DOMAIN_BOUNDS[1], config.DOMAIN_BOUNDS[1], config.DOMAIN_BOUNDS[1], time_interval[1]])

    if use_lhs:
        sampler = LatinHypercube(d=dim, seed=config.SEED + int(time_interval[0] * 100))
        points = lb + (ub - lb) * sampler.random(n=N)
    else:
        points = lb + (ub - lb) * np.random.rand(N, dim)

    return torch.tensor(points, dtype=torch.float32, device=config.DEVICE, requires_grad=requires_grad)


def sample_initial(N, t_start, requires_grad=False, use_lhs=True):
    """Samples N initial condition points at t=t_start."""
    dim = 3  # x, y, z
    lb = np.array([config.DOMAIN_BOUNDS[0]] * dim)
    ub = np.array([config.DOMAIN_BOUNDS[1]] * dim)

    if use_lhs:
        sampler = LatinHypercube(d=dim, seed=config.SEED + int(t_start * 100) + 1)
        points_xyz = lb + (ub - lb) * sampler.random(n=N)
    else:
        points_xyz = lb + (ub - lb) * np.random.rand(N, dim)

    points_t = np.full((N, 1), t_start)
    points = np.hstack((points_xyz, points_t))
    return torch.tensor(points, dtype=torch.float32, device=config.DEVICE, requires_grad=requires_grad)


def sample_boundary(N_total, time_interval, requires_grad=False, use_lhs=True):
    """Samples points on the 6 faces of the 3D domain."""
    N_per_face = N_total // 6
    if N_per_face == 0: N_per_face = 1

    lb = np.array([config.DOMAIN_BOUNDS[0], config.DOMAIN_BOUNDS[0], config.DOMAIN_BOUNDS[0], time_interval[0]])
    ub = np.array([config.DOMAIN_BOUNDS[1], config.DOMAIN_BOUNDS[1], config.DOMAIN_BOUNDS[1], time_interval[1]])

    all_points = []
    base_seed = config.SEED + int(time_interval[0] * 100) + 2

    # Helper to sample one face
    def sample_face(N, dim_indices, fixed_dim, fixed_val, seed):
        sampler_dims = 3
        if use_lhs:
            varying_points = LatinHypercube(d=sampler_dims, seed=seed).random(n=N)
        else:
            np.random.seed(seed);
            varying_points = np.random.rand(N, sampler_dims)

        varying_points = lb[dim_indices] + (ub[dim_indices] - lb[dim_indices]) * varying_points
        face_points = np.zeros((N, 4))
        face_points[:, dim_indices] = varying_points
        face_points[:, fixed_dim] = fixed_val
        return face_points

    # Sample each face and store in order: x-, x+, y-, y+, z-, z+
    all_points.append(sample_face(N_per_face, [1, 2, 3], 0, lb[0], base_seed + 0))  # x=0
    all_points.append(sample_face(N_per_face, [1, 2, 3], 0, ub[0], base_seed + 1))  # x=1
    all_points.append(sample_face(N_per_face, [0, 2, 3], 1, lb[1], base_seed + 2))  # y=0
    all_points.append(sample_face(N_per_face, [0, 2, 3], 1, ub[1], base_seed + 3))  # y=1
    all_points.append(sample_face(N_per_face, [0, 1, 3], 2, lb[2], base_seed + 4))  # z=0
    all_points.append(sample_face(N_per_face, [0, 1, 3], 2, ub[2], base_seed + 5))  # z=1

    final_points = np.vstack(all_points)

    # Add extra points if N_total is not divisible by 6
    num_missing = N_total - final_points.shape[0]
    if num_missing > 0:
        extra_points = sample_face(num_missing, [1, 2, 3], 0, lb[0], base_seed + 6)
        final_points = np.vstack([final_points, extra_points])

    return torch.tensor(final_points, dtype=torch.float32, device=config.DEVICE, requires_grad=requires_grad)