# main.py

import torch
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io

# Import our custom modules
import config
from networks import FNO3d_onestep
from functions import get_grid3d_from_config, initial_condition_grid, fno_onestep_loss
from post_processing import plot_results_slice_from_trajectory, save_results_matlab_from_trajectory


def main():
    # ... (Setup code is the same) ...
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    model = FNO3d_onestep(
        modes1=config.MODES, modes2=config.MODES, modes3=config.MODES,
        width=config.WIDTH, n_layers=config.N_LAYERS
    ).to(device)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    grid = get_grid3d_from_config(device)
    u0 = initial_condition_grid(grid)
    u0_batch = u0.unsqueeze(0).repeat(config.BATCH_SIZE, 1, 1, 1, 1)

    # --- Training Loop with Debug Prints ---
    print(f"Starting training for {config.EPOCHS} epochs...")
    start_time_total = time.time()

    for epoch in tqdm(range(config.EPOCHS), desc="Epochs"):
        model.train()
        epoch_loss = 0.0
        # --- Add accumulators for debug values ---
        epoch_ut_mse = 0.0
        epoch_muspatial_mse = 0.0

        u_current = u0_batch

        for _ in range(config.TOTAL_TIME_STEPS):
            optimizer.zero_grad()
            u_pred = model(u_current)

            # --- Unpack the loss and debug values ---
            loss, debug_ut, debug_muspatial = fno_onestep_loss(u_current, u_pred, grid)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ut_mse += debug_ut.item()
            epoch_muspatial_mse += debug_muspatial.item()

            u_current = u_pred.detach()

        scheduler.step()

        avg_epoch_loss = epoch_loss / config.TOTAL_TIME_STEPS
        avg_ut_mse = epoch_ut_mse / config.TOTAL_TIME_STEPS
        avg_muspatial_mse = epoch_muspatial_mse / config.TOTAL_TIME_STEPS

        if (epoch + 1) % 2 == 0:
            # --- THE NEW, DETAILED PRINT STATEMENT ---
            tqdm.write(f"Epoch {epoch + 1}/{config.EPOCHS} | "
                       f"Total Loss: {avg_epoch_loss:.3e} | "
                       f"u_t Term: {avg_ut_mse:.3e} | "
                       f"μ_spatial Term: {avg_muspatial_mse:.3e} | "
                       f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # ... (Rest of main.py is the same) ...
    print(f"\nTotal Training Time: {time.time() - start_time_total:.2f} seconds")
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, "fno_final_model.pth"))

    print("\nGenerating final trajectory...")
    model.eval()
    with torch.no_grad():
        full_trajectory_list = [u0]
        u_current = u0.unsqueeze(0)
        for _ in range(config.TOTAL_TIME_STEPS):
            u_pred = model(u_current)
            full_trajectory_list.append(u_pred.squeeze(0))
            u_current = u_pred
        final_trajectory_cpu = torch.stack(full_trajectory_list, dim=-2).cpu()

    plot_results_slice_from_trajectory(final_trajectory_cpu, config.PLOT_TIMES)
    save_results_matlab_from_trajectory(final_trajectory_cpu, config.MATLAB_SAVE_TIMES)


if __name__ == '__main__':
    main()