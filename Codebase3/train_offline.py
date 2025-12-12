# train_offline.py
import torch
import torch.nn as nn
import numpy as np
import pickle
import sys
import time
from sklearn.preprocessing import MinMaxScaler

import config
import data_loader
from ai_models import WindBiLSTM


# Helper to create time-series windows
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


if __name__ == "__main__":
    start_time = time.time()

    print("\n" + "=" * 50)
    print("      AI TRAINING ENGINE (BiLSTM) INITIALIZED")
    print("=" * 50)

    # --- 1. Data Loading ---
    print("\n[1/4] Generating Synthetic Data...")
    df = data_loader.generate_refinery_data()
    data = df['Speed'].values.reshape(-1, 1)
    print(f"      -> Loaded {len(df)} timestamps.")

    # --- 2. Preprocessing ---
    print("[2/4] Normalizing & Sequencing...")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Save scaler (Crucial for visualizer to understand the model later)
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    X, y = create_sequences(data_scaled, config.SEQ_LENGTH)

    # Convert to Tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    # Split 80/20
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"      -> Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")

    # --- 3. Training Loop ---
    print(f"\n[3/4] Training BiLSTM Network ({config.EPOCHS} Epochs)...")
    model = WindBiLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.train()

    # VISIBILITY: Progress Bar
    for epoch in range(config.EPOCHS):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # Calculate roughly how much % is done
        progress = (epoch + 1) / config.EPOCHS
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        # Print metrics on the same line
        sys.stdout.write(f'\r      Epoch {epoch + 1:02d}/{config.EPOCHS} |{bar}| Loss: {loss.item():.6f}')
        sys.stdout.flush()

    print("\n      -> Training Complete.")

    # --- 4. Validation & Saving ---
    print("\n[4/4] Validating & Saving...")
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test)

    print(f"      -> Final Test MSE: {test_loss.item():.6f}")
    if test_loss.item() < 0.05:
        print("      -> STATUS: Model is HIGHLY ACCURATE.")
    else:
        print("      -> STATUS: Model is CONVERGING (Acceptable).")

    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"      -> Saved Model to: {config.MODEL_PATH}")

    total_time = time.time() - start_time
    print(f"\nTotal Runtime: {total_time:.2f} seconds")
    print("=" * 50 + "\n")