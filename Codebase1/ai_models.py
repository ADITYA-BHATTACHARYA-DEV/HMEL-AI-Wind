# # ai_models.py
# from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# import config
#
#
# class WindPatternAI:
#     def __init__(self):
#         # We look for 3 regimes: Low (Calm), Medium (Base), High (Storm/Power)
#         self.model = GaussianMixture(n_components=config.CLUSTERS, random_state=config.RANDOM_SEED)
#         self.scaler = StandardScaler()
#
#     def find_regimes(self, df):
#         print("Running AI Pattern Recognition...")
#
#         # We feed the AI: Speed, Power, and Direction Vectors
#         features = df[['Speed', 'WPD', 'Wx', 'Wy']]
#
#         # Normalize data (AI handles standard distributions better)
#         X_scaled = self.scaler.fit_transform(features)
#
#         # Predict Clusters
#         df['Cluster'] = self.model.fit_predict(X_scaled)
#
#         # Auto-Label the Clusters based on average Power Density
#         cluster_stats = df.groupby('Cluster')['WPD'].mean().sort_values()
#
#         labels = {
#             cluster_stats.index[0]: 'Low Potential',
#             cluster_stats.index[1]: 'Base Load',
#             cluster_stats.index[2]: 'High Power'
#         }
#
#         df['Regime_Label'] = df['Cluster'].map(labels)
#         return df

# ai_models.py
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import config


class WindPatternAI:
    def __init__(self):
        # We look for 3 regimes: Low (Calm), Medium (Base), High (Storm/Power)
        self.model = GaussianMixture(n_components=config.CLUSTERS, random_state=config.RANDOM_SEED)
        self.scaler = StandardScaler()

    def simulate_turbines(self, df):
        """
        Simulates Energy Yield (kWh) for different turbine technologies
        based on the site's specific wind/density profile.
        """
        print("   -> Running Virtual Turbine Simulation...")

        # --- Turbine Tech 1: Vertical Axis (VAWT) ---
        # Good for low speed/turbulence. Starts at 1.5 m/s.
        def vawt_efficiency(speed):
            if speed < 1.5: return 0.0
            if speed > 18: return 0.0  # Cut-out
            # Simple efficiency curve peaking at 0.35 (Betz limit is 0.59)
            return min(0.35, (speed - 1.5) / 10)

            # --- Turbine Tech 2: Small Horizontal (HAWT) ---

        # Standard type. Starts at 3.0 m/s (Needs higher wind).
        def hawt_efficiency(speed):
            if speed < 3.0: return 0.0
            if speed > 25: return 0.0
            return min(0.45, (speed - 3.0) / 9)

            # Time duration per row (assuming 15 min data = 0.25h)

        hours = 0.25

        # Calculate Energy: Power (W) * Time (h) / 1000 = kWh
        # Power = 0.5 * Rho * Area * V^3 * Efficiency
        # We assume 1 m^2 swept area for fair comparison

        # VAWT uses ground speed (10m)
        df['Energy_VAWT_kWh'] = df.apply(lambda row:
                                         (0.5 * row['Rho'] * 1 * (row['Speed_10m'] ** 3) * vawt_efficiency(
                                             row['Speed_10m'])) * hours / 1000, axis=1)

        # HAWT uses tower speed (30m) because you'd mount it higher
        df['Energy_HAWT_30m_kWh'] = df.apply(lambda row:
                                             (0.5 * row['Rho'] * 1 * (row['Speed_30m'] ** 3) * hawt_efficiency(
                                                 row['Speed_30m'])) * hours / 1000, axis=1)

        return df

    def find_regimes(self, df):
        print("Running AI Pattern Recognition & Simulation...")

        # We feed the AI: Speed, Power, and Direction Vectors
        features = df[['Speed', 'WPD', 'Wx', 'Wy']]

        # Normalize data (AI handles standard distributions better)
        X_scaled = self.scaler.fit_transform(features)

        # Predict Clusters
        df['Cluster'] = self.model.fit_predict(X_scaled)

        # Auto-Label the Clusters based on average Power Density
        cluster_stats = df.groupby('Cluster')['WPD'].mean().sort_values()

        labels = {
            cluster_stats.index[0]: 'Low Potential',
            cluster_stats.index[1]: 'Base Load',
            cluster_stats.index[2]: 'High Power'
        }

        df['Regime_Label'] = df['Cluster'].map(labels)

        # NEW: Run the Simulation
        df = self.simulate_turbines(df)

        return df