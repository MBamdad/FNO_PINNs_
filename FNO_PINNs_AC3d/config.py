# config.py

import numpy as np
import torch

# --- Core Settings ---
SEED = 42
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# --- PHYSICAL PARAMETERS (Now used directly in the loss) ---
LAMBDA_PARAM = 10.0
EPSILON_PARAM = 0.025
DROP_RADIUS_PARAM = 0.35
DROP_CENTER_X = 0.5
DROP_CENTER_Y = 0.5
DROP_CENTER_Z = 0.5

# --- FNO & Training Hyperparameters ---
GRID_RESOLUTION = 32
# --- Back to PHYSICAL TIME ---
TIME_END = 5.0
DT = 0.02  # Use a smaller physical time step for stability with the full PDE
TOTAL_TIME_STEPS = int(TIME_END / DT)

# FNO Architecture
MODES = 12
WIDTH = 32
N_LAYERS = 4

# Training
EPOCHS = 100 # 50 # 300 # Let's try with 300 epochs
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# --- Post-processing and Saving ---
MODEL_SAVE_DIR = "saved_models_ac3d_final_100Epochs" # EPOCHS = 100 #
PLOT_TIMES = np.array([0.0, 1.5, 2.5, 5.0])
MATLAB_SAVE_TIMES = np.array([0.0, 1.5, 2.5, 5.0])
PLOT_Z_SLICE = 0.5
MATLAB_SAVE_FILENAME = "ac3d_final_results_100Epochs.mat" # EPOCHS = 100 #
MATLAB_SAVE_RESOLUTION = GRID_RESOLUTION