import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from catboost import CatBoostRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import config


# --- PART 1: BiLSTM WITH TEMPORAL ATTENTION ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.v = nn.Linear(hidden_size * 2, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_dim * 2]
        # encoder_outputs: [batch_size, seq_len, hidden_dim * 2]
        seq_len = encoder_outputs.size(1)

        # Repeat hidden state seq_len times
        hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Calculate energy (attention scores)
        energy = torch.tanh(self.attn(encoder_outputs))
        attention_scores = self.v(energy).squeeze(2)

        # Softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted sum of encoder outputs
        context_vector = (encoder_outputs * attention_weights.unsqueeze(2)).sum(dim=1)
        return context_vector, attention_weights


class WindBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, layers=2):
        super(WindBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers,
                            batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, 1)  # Output 1 value (Speed)

    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, (hn, cn) = self.lstm(x)

        # Use Attention instead of just the last hidden state
        # We concatenate the forward and backward hidden states for the query
        final_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        context_vector, _ = self.attention(final_hidden, lstm_out)

        prediction = self.fc(context_vector)
        return prediction


# --- PART 2: THE HYBRID CONTROLLER ---
class HybridWindModel:
    def __init__(self):
        self.lstm_model = WindBiLSTM(input_size=1, hidden_size=64, layers=2)
        self.catboost_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            verbose=False,
            allow_writing_files=False
        )
        self.scaler = None

    def train_hybrid(self, X_seq, y_target, X_tabular):
        """
        X_seq: Tensor for BiLSTM [Batch, Seq, 1]
        y_target: Tensor for Targets
        X_tabular: DataFrame for CatBoost (Context Features)
        """
        # 1. Train BiLSTM
        print("   -> [Layer 1] Training BiLSTM + Attention...")
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        self.lstm_model.train()

        # Simple training loop (for brevity, assuming pre-batched or full batch)
        for epoch in range(50):
            optimizer.zero_grad()
            out = self.lstm_model(X_seq)
            loss = criterion(out, y_target)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0: print(f"      Epoch {epoch}: Loss {loss.item():.4f}")

        # 2. Get BiLSTM Predictions on Training Data
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_preds = self.lstm_model(X_seq).numpy()

        # 3. Calculate Residuals (Errors)
        # Residual = Actual - BiLSTM_Prediction
        y_numpy = y_target.numpy()
        residuals = y_numpy - lstm_preds

        # 4. Train CatBoost on Residuals
        # CatBoost sees: [Temp, Humidity, Hour, Lag...] -> Predicts: [Error]
        print("   -> [Layer 2] Training CatBoost Residual Corrector...")
        self.catboost_model.fit(X_tabular, residuals)
        print("   -> Hybrid Training Complete.")

    def predict(self, X_seq, X_tabular):
        # 1. BiLSTM Prediction (The Trend)
        self.lstm_model.eval()
        with torch.no_grad():
            base_pred = self.lstm_model(X_seq).numpy()

        # 2. CatBoost Prediction (The Correction)
        correction = self.catboost_model.predict(X_tabular)

        # 3. Combine
        # Reshape correction to match base_pred if needed
        correction = correction.reshape(-1, 1)
        final_pred = base_pred + correction

        return final_pred

    def save(self, path_lstm, path_cb):
        torch.save(self.lstm_model.state_dict(), path_lstm)
        self.catboost_model.save_model(path_cb)

    def load(self, path_lstm, path_cb):
        self.lstm_model.load_state_dict(torch.load(path_lstm))
        self.lstm_model.eval()
        self.catboost_model.load_model(path_cb)