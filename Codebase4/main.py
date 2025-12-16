import sys
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import data_loader
import visualizer
import config


def add_regime_labels(df):
    """
    Restores the clustering logic needed for Dashboard Plot 5 (Regime Stability).
    Previously this was in 'WindPatternAI', now we do it here.
    """
    # 1. Select features for clustering
    data = df[['Speed', 'WPD']].dropna()

    if len(data) == 0:
        return df  # Safety catch for empty data

    # 2. Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # 3. Cluster (Low, Medium, High winds)
    gmm = GaussianMixture(n_components=3, random_state=42)
    clusters = gmm.fit_predict(X_scaled)

    # 4. Align clusters back to original dataframe indices
    df.loc[data.index, 'Cluster'] = clusters

    # 5. Auto-Label: Sort clusters by mean wind speed to name them consistently
    cluster_means = df.groupby('Cluster')['Speed'].mean().sort_values()

    # Map: Lowest Mean -> "Low Potential", Medium -> "Base Load", Highest -> "High Power"
    label_map = {
        cluster_means.index[0]: 'Low Potential',
        cluster_means.index[1]: 'Base Load',
        cluster_means.index[2]: 'High Power'
    }
    df['Regime_Label'] = df['Cluster'].map(label_map)

    # 6. Calculate Energy ROI columns (Needed for Plot 4)
    if 'Energy_VAWT_kWh' not in df.columns:
        hours = 0.25  # 15 min steps
        # VAWT Efficiency ~30%
        df['Energy_VAWT_kWh'] = (0.5 * df['Rho'] * (df['Speed'] ** 3) * 0.3 * 1) * hours / 1000
        # HAWT Efficiency ~40% (at 30m)
        df['Energy_HAWT_30m_kWh'] = (0.5 * df['Rho'] * (df['Speed_30m'] ** 3) * 0.4 * 1) * hours / 1000

    return df


def main():
    try:
        print("╔══════════════════════════════════════════╗")
        print("║   STARTING WIND ASSESSMENT PIPELINE      ║")
        print("╚══════════════════════════════════════════╝")

        # Step 1: Load
        print("\n--- Step 1: Data Ingestion ---")
        raw_data = data_loader.load_data()

        # Diagnostics
        print(f"   -> Data Range: {raw_data['Timestamp'].min()} to {raw_data['Timestamp'].max()}")
        print(f"   -> Total Rows: {len(raw_data)}")

        # Step 2: Clustering & Physics
        print("\n--- Step 2: Regime Clustering & Physics ---")
        # Replaces the old 'ai_engine.find_regimes()' call
        processed_data = add_regime_labels(raw_data)
        print("   -> Regimes Identified & ROI Calculated.")

        # Step 3: Visualization & Strategy
        # The visualizer will internally call the NEW HybridWindModel for predictions
        print("\n--- Step 3: Reporting & Dashboard ---")
        visualizer.print_summary(processed_data)
        visualizer.plot_dashboard(processed_data)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()