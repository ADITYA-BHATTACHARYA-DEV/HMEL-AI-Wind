import torch
import numpy as np
import pickle
import sys
from tqdm import tqdm  # Import the progress bar library
import data_loader
from ai_models import HybridWindModel
import config


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   HYBRID TRAINING ENGINE (BiLSTM-Attention + CatBoost)       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # 1. Load & Prep
    print("\n[1/4] Loading & Preprocessing Data...")
    try:
        df = data_loader.load_data()
        X_seq, X_tab, y, scaler = data_loader.prepare_hybrid_tensors(df, config.SEQ_LENGTH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Save Scaler
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print("      -> Data Loaded. Scaler saved.")

    # Split
    split = int(len(X_seq) * 0.8)
    X_seq_train, X_seq_test = X_seq[:split], X_seq[split:]
    X_tab_train, X_tab_test = X_tab[:split], X_tab[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"      -> Training Samples: {len(X_seq_train)} | Test Samples: {len(X_seq_test)}")

    # 2. Initialize Model
    model = HybridWindModel()

    # --- CUSTOM TRAINING LOOP WITH PROGRESS BAR ---
    print("\n[2/4] Training Layer 1: BiLSTM-Attention Network")

    # We override the internal train loop here to show the bar
    # (Assuming we modify the logic slightly or just wrap the existing call if it was exposed)
    # Since HybridWindModel.train_hybrid usually hides the loop, let's wrap the *epochs* if possible.
    # However, since train_hybrid is a single call, we will add a simulated loading bar
    # OR better yet, we can modify ai_models.py to accept a callback.

    # For now, let's rely on modifying ai_models.py slightly or
    # just wrap the high-level steps if we can't touch ai_models.

    # NOTE: To get a REAL epoch-by-epoch bar, we need to edit ai_models.py.
    # Below is the logic assuming you updated ai_models.py as shown in the next block.
    # If not, this simple wrapper just shows "Working..."

    # Let's perform the training call.
    # To make this truly interactive, I will assume you update the training function
    # to be iterable or we just show a bar for the Steps.

    with tqdm(total=2, desc="   -> Hybrid Stages", unit="stage") as pbar:
        # Stage A: BiLSTM
        pbar.set_description("   -> Training BiLSTM (Layer 1)")

        # We define a custom training loop here instead of calling model.train_hybrid
        # to give you granular control over the progress bar.

        optimizer = torch.optim.Adam(model.lstm_model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        model.lstm_model.train()

        EPOCHS = 50
        # Nested progress bar for Epochs
        for epoch in tqdm(range(EPOCHS), desc="      Epochs", leave=False):
            optimizer.zero_grad()
            out = model.lstm_model(X_seq_train)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()

        pbar.update(1)  # BiLSTM Done

        # Stage B: CatBoost
        pbar.set_description("   -> Training CatBoost (Layer 2)")

        # Get Residuals
        model.lstm_model.eval()
        with torch.no_grad():
            lstm_preds = model.lstm_model(X_seq_train).numpy()

        y_numpy = y_train.numpy()
        residuals = y_numpy - lstm_preds

        # Fit CatBoost (It has its own internal verbose, but we silence it for our bar)
        model.catboost_model.fit(X_tab_train, residuals, verbose=False)

        pbar.update(1)  # CatBoost Done

    print("\n[3/4] Evaluation & Validation...")
    preds = model.predict(X_seq_test, X_tab_test)
    preds_actual = scaler.inverse_transform(preds)
    y_actual = scaler.inverse_transform(y_test.numpy())

    rmse = np.sqrt(np.mean((preds_actual - y_actual) ** 2))
    print(f"      -> Final Hybrid RMSE: {rmse:.4f} m/s")

    # 4. Save
    print("\n[4/4] Saving Artifacts...")
    model.save("lstm_attn.pth", "catboost_resid.cbm")
    print("      -> Models saved to disk.")
    print("\n[SUCCESS] Pipeline Complete.")


if __name__ == "__main__":
    main()