# forecaster.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class WindBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(WindBiLSTM, self).__init__()
        # Bidirectional=True makes it a BiLSTM
        # It learns from the past (Forward pass) and future context (Backward pass) during training
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

        # The output layer (times 2 because BiLSTM doubles the features)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        lstm_out, _ = self.lstm(x)

        # We only care about the last time step for prediction
        last_time_step = lstm_out[:, -1, :]
        prediction = self.fc(last_time_step)
        return prediction


class ForecastEngine:
    def __init__(self):
        self.model = WindBiLSTM()
        self.scaler = MinMaxScaler()
        self.seq_length = 24  # Look back 24 hours (assuming hourly data)

    def prepare_sequences(self, data):
        """Converts linear data into time-sequences [t-24 ... t] -> [t+1]"""
        sequences = []
        targets = []
        data = self.scaler.fit_transform(data.reshape(-1, 1))

        for i in range(len(data) - self.seq_length):
            seq = data[i:i + self.seq_length]
            target = data[i + self.seq_length]
            sequences.append(seq)
            targets.append(target)

        return torch.FloatTensor(np.array(sequences)), torch.FloatTensor(np.array(targets))

    def train_forecast(self, df_speed):
        print("Training BiLSTM for Time-Series Forecasting...")

        # Prepare Data
        X, y = self.prepare_sequences(df_speed.values)

        # Training Loop (Simplified)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(50):  # 50 Epochs
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        print(f"-> Training Complete. Final Loss: {loss.item():.4f}")

    def predict_next_24h(self, current_history):
        """Predicts the wind speed for the NEXT timestep"""
        self.model.eval()
        with torch.no_grad():
            # Scale input
            scaled_hist = self.scaler.transform(current_history.reshape(-1, 1))
            tensor_hist = torch.FloatTensor(scaled_hist).unsqueeze(0)  # Add batch dim

            # Predict
            pred_scaled = self.model(tensor_hist)
            pred_actual = self.scaler.inverse_transform(pred_scaled.numpy())

            return pred_actual[0][0]