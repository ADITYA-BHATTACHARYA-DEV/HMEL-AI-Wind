# config.py

# --- EXISTING CONFIG ---
CLUSTERS = 3
RANDOM_SEED = 42

# --- NEW ADDITIONS (For BiLSTM) ---
import torch
MODEL_PATH = 'wind_bilstm.pth'
SCALER_PATH = 'scaler.pkl'
SEQ_LENGTH = 24  # Look back 24 steps (6 hours)
HIDDEN_SIZE = 64
LAYERS = 2
EPOCHS = 50
LEARNING_RATE = 0.001