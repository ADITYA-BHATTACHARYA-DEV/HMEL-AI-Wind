# ai_models.py
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
import config


class WindPatternAI:
    def __init__(self):
        # We look for 3 regimes: Low (Calm), Medium (Base), High (Storm/Power)
        self.model = GaussianMixture(n_components=config.CLUSTERS, random_state=config.RANDOM_SEED)
        self.scaler = StandardScaler()

    def find_regimes(self, df):
        print("Running AI Pattern Recognition...")

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
        return df