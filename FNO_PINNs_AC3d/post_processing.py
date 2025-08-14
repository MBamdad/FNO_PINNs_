# post_processing.py

import os
import numpy as np
import torch
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io
import config


def plot_results_slice_from_trajectory(trajectory, plot_times, save_filename='AC3D_Final_Slice.png'):
    """
    Generates a 2D slice plot from the full trajectory in physical time.
    """
    print("\nGenerating 2D slice plots...")
    S = trajectory.shape[0]
    x = np.linspace(0, 1, S)
    y = np.linspace(0, 1, S)
    X, Y = np.meshgrid(x, y, indexing='ij')

    fig, axes = plt.subplots(1, len(plot_times), figsize=(20, 5), squeeze=False)
    axes = axes.flatten()

    for i, t_phys in enumerate(plot_times):
        ax = axes[i]

        # Direct conversion from physical time to index
        time_index = int(round(t_phys / config.DT))

        if time_index < trajectory.shape[-2]:
            u_field = trajectory.numpy()
            z_slice_idx = S // 2
            U_slice = u_field[:, :, z_slice_idx, time_index].squeeze()

            im = ax.imshow(U_slice.T, origin='lower', extent=[0, 1, 0, 1],
                           cmap='viridis', vmin=-1.0, vmax=1.0)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f't={t_phys:.2f}, z={config.PLOT_Z_SLICE:.2f}')
            ax.set_xlabel('x');
            ax.set_ylabel('y')
        else:
            ax.set_title(f't={t_phys:.2f} (Out of bounds)')
            print(f"Warning: Time t={t_phys:.2f} (index {time_index}) is out of bounds for plotting.")

    plt.tight_layout(pad=1.5)
    plt.savefig(save_filename, dpi=300)
    print(f"Saved plot to {save_filename}")
    plt.show()


def save_results_matlab_from_trajectory(trajectory, save_times):
    """
    Saves results from the trajectory to a .mat file in physical time.
    """
    print("\nSaving results for MATLAB...")
    S = trajectory.shape[0]

    x_vec = np.linspace(0, 1, S)
    y_vec = np.linspace(0, 1, S)
    z_vec = np.linspace(0, 1, S)
    X_mg, Y_mg, Z_mg = np.meshgrid(x_vec, y_vec, z_vec, indexing='xy')

    U_save_list = []
    times_saved = []

    for t_phys in save_times:
        time_index = int(round(t_phys / config.DT))

        if time_index < trajectory.shape[-2]:
            U_3d_tensor = trajectory[..., time_index, 0]
            U_3d = U_3d_tensor.permute(1, 0, 2).numpy()
            U_save_list.append(U_3d)
            times_saved.append(t_phys)
        else:
            print(f"Warning: Time t={t_phys:.4f} (index {time_index}) is out of bounds for saving.")

    if not U_save_list:
        print("Error: No data to save to .mat file.")
        return

    U_save = np.stack(U_save_list, axis=-1)

    mat_data = {
        'X_mg': X_mg, 'Y_mg': Y_mg, 'Z_mg': Z_mg,
        'T': np.array(times_saved),
        'U': U_save
    }
    scipy.io.savemat(config.MATLAB_SAVE_FILENAME, mat_data, do_compression=True)
    print(f"Successfully saved results to {config.MATLAB_SAVE_FILENAME}")