# ai_models.py
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import config


# ==========================================
# 1. EXISTING CLASS: CLUSTERING & PHYSICS
# ==========================================
class WindPatternAI:
    def __init__(self):
        # We look for 3 regimes: Low (Calm), Medium (Base), High (Storm/Power)
        self.model = GaussianMixture(n_components=config.CLUSTERS, random_state=config.RANDOM_SEED)
        self.scaler = StandardScaler()

    def simulate_turbines(self, df):
        """
        Simulates Energy Yield (kWh) for different turbine technologies.
        """
        print("   -> Running Virtual Turbine Simulation...")

        # --- Turbine Tech 1: Vertical Axis (VAWT) ---
        def vawt_efficiency(speed):
            if speed < 1.5: return 0.0
            if speed > 18: return 0.0
            return min(0.35, (speed - 1.5) / 10)

        # --- Turbine Tech 2: Small Horizontal (HAWT) ---
        def hawt_efficiency(speed):
            if speed < 3.0: return 0.0
            if speed > 25: return 0.0
            return min(0.45, (speed - 3.0) / 9)

        hours = 0.25

        # VAWT uses ground speed (10m)
        df['Energy_VAWT_kWh'] = df.apply(lambda row:
                                         (0.5 * row['Rho'] * 1 * (row['Speed_10m'] ** 3) * vawt_efficiency(
                                             row['Speed_10m'])) * hours / 1000, axis=1)

        # HAWT uses tower speed (30m)
        df['Energy_HAWT_30m_kWh'] = df.apply(lambda row:
                                             (0.5 * row['Rho'] * 1 * (row['Speed_30m'] ** 3) * hawt_efficiency(
                                                 row['Speed_30m'])) * hours / 1000, axis=1)
        return df

    def find_regimes(self, df):
        print("Running AI Pattern Recognition & Simulation...")
        features = df[['Speed', 'WPD', 'Wx', 'Wy']]
        X_scaled = self.scaler.fit_transform(features)

        # Predict Clusters
        df['Cluster'] = self.model.fit_predict(X_scaled)

        # Auto-Label
        cluster_stats = df.groupby('Cluster')['WPD'].mean().sort_values()
        labels = {
            cluster_stats.index[0]: 'Low Potential',
            cluster_stats.index[1]: 'Base Load',
            cluster_stats.index[2]: 'High Power'
        }
        df['Regime_Label'] = df['Cluster'].map(labels)

        # Run Simulation
        df = self.simulate_turbines(df)
        return df


# ==========================================
# 2. NEW ADDITION: FORECASTING ENGINE
# ==========================================
class WindBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=config.HIDDEN_SIZE, output_size=1):
        super(WindBiLSTM, self).__init__()

        # Bidirectional LSTM: Learns past (forward) and future context (backward)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=config.LAYERS,
            batch_first=True,
            bidirectional=True
        )

        # Output layer (hidden_size * 2 because of bidirectionality)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        out, _ = self.lstm(x)
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        prediction = self.fc(out)
        return prediction