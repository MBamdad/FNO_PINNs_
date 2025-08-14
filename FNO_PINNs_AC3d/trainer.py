# trainer.py

import torch
import time
import config
from dataset import sample_initial, sample_boundary, sample_domain
from functions import initial_condition_3d, compute_loss_3d


def train_interval(model, t_start, t_end, prev_model, device):
    """
    Trains a PINN model for a single time interval [t_start, t_end].
    """
    print(f"\n--- Training Interval [{t_start:.2f}, {t_end:.2f}] ---")
    is_first_interval = (prev_model is None)
    current_C0 = config.C0_INITIAL if is_first_interval else config.C0_SUBSEQUENT
    print(f"Using C0 weight = {current_C0}")

    # 1. Generate Training Data
    xyz_u = sample_initial(config.N_U, t_start)
    if is_first_interval:
        with torch.no_grad():
            u_target = initial_condition_3d(xyz_u[:, 0:1], xyz_u[:, 1:2], xyz_u[:, 2:3])
    else:
        prev_model.eval()
        with torch.no_grad():
            # Ensure prev_model is on the correct device for inference
            out_prev = prev_model.to(device)(xyz_u)
            u_target = out_prev[:, 0:1]

    xyzt_b = sample_boundary(config.N_B, [t_start, t_end])
    xyzt_f = sample_domain(config.N_F, [t_start, t_end], requires_grad=True)

    # 2. Adam Optimization
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=config.ADAM_LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_adam, gamma=0.9995)
    model.train()
    start_time_adam = time.time()

    print("Starting Adam optimization...")
    for epoch in range(config.ADAM_EPOCHS):
        optimizer_adam.zero_grad()
        loss, mse_u, mse_b, mse_f = compute_loss_3d(model, xyz_u, u_target, xyzt_b, xyzt_f, current_C0, device)
        if torch.isnan(loss):
            print(f"NaN loss at Adam epoch {epoch}. Stopping interval training.")
            return False  # Failure
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()
        if epoch > 0 and epoch % 100 == 0: scheduler.step()
        if epoch % 1000 == 0:
            print(f'Adam Ep {epoch:>5}: Loss {loss.item():.4e}, MSE_u {mse_u.item():.3e}, '
                  f'MSE_b {mse_b.item():.3e}, MSE_f {mse_f.item():.3e}')

    print(f"Adam finished in {time.time() - start_time_adam:.2f}s.")

    # 3. L-BFGS Optimization
    print("Starting L-BFGS optimization...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), max_iter=config.LBFGS_MAX_ITER, line_search_fn="strong_wolfe",
        tolerance_grad=1e-9, tolerance_change=1e-10, history_size=100
    )
    closure_count = [0]  # Use a list to make it mutable inside closure

    def closure():
        optimizer_lbfgs.zero_grad()
        loss, mse_u, mse_b, mse_f = compute_loss_3d(model, xyz_u, u_target, xyzt_b, xyzt_f, current_C0, device)
        if torch.isnan(loss): return loss
        loss.backward()
        if closure_count[0] % 500 == 0:
            print(f'LBFGS Eval {closure_count[0]:>5}: Loss {loss.item():.4e}')
        closure_count[0] += 1
        return loss

    start_time_lbfgs = time.time()
    try:
        optimizer_lbfgs.step(closure)
    except Exception as e:
        print(f"L-BFGS optimization failed: {e}")
        return False  # Failure

    print(f"L-BFGS finished in {time.time() - start_time_lbfgs:.2f}s.")

    final_loss, _, _, _ = compute_loss_3d(model, xyz_u, u_target, xyzt_b, xyzt_f, current_C0, device)
    if torch.isnan(final_loss):
        print("Final loss is NaN. Interval training failed.")
        return False

    print(f'Final Loss: {final_loss.item():.4e}')
    return True  # Success