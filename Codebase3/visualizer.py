# visualizer.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import torch
import pickle
from sklearn.metrics import mean_squared_error, r2_score

import config
from ai_models import WindBiLSTM


# --- NEW HELPER: LOAD & PREDICT ---
def load_and_predict(df):
    try:
        # Load Model
        model = WindBiLSTM()
        model.load_state_dict(torch.load(config.MODEL_PATH))
        model.eval()

        # Load Scaler
        with open(config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)

        print("   -> AI Model Loaded. Generating Forecasts...")

        # Prepare Data
        data = df['Speed'].values.reshape(-1, 1)
        data_scaled = scaler.transform(data)

        # Generate Sequences
        X = []
        for i in range(len(data_scaled) - config.SEQ_LENGTH):
            X.append(data_scaled[i:(i + config.SEQ_LENGTH)])

        # Predict
        X_tensor = torch.FloatTensor(np.array(X))
        with torch.no_grad():
            preds_scaled = model(X_tensor).numpy()

        preds_actual = scaler.inverse_transform(preds_scaled)

        # Align with DataFrame (Pad the beginning)
        padding = np.full((config.SEQ_LENGTH, 1), np.nan)
        full_preds = np.vstack([padding, preds_actual])
        df['Predicted_Speed'] = full_preds

        return df, True  # Success flag

    except FileNotFoundError:
        print("   -> (!) AI Model not found. Skipping prediction layer.")
        df['Predicted_Speed'] = np.nan
        return df, False


# --- EXISTING DASHBOARD FUNCTION ---
def plot_dashboard(df):
    # 1. Run Predictions first
    df, has_predictions = load_and_predict(df)

    print("Generating Advanced Dashboard...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Increased figure size for the extra row
    fig = plt.figure(figsize=(20, 16))

    # CHANGED: 3 Rows now instead of 2
    gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[1, 1, 1])

    fig.suptitle('Advanced Wind Energy Site Assessment (AI Enhanced)', fontsize=20, fontweight='bold', y=0.95)

    # --- ROW 1 (EXISTING) ---

    # 1. Vertical Profile
    ax1 = fig.add_subplot(gs[0, 0:2])
    speeds = [df['Speed_10m'].mean(), df['Speed_30m'].mean(), df['Speed_50m'].mean()]
    heights = [10, 30, 50]
    ax1.plot(speeds, heights, linestyle='--', color='gray', alpha=0.5)
    ax1.plot(speeds, heights, marker='o', linewidth=0, color='crimson', markersize=10)
    ax1.set_title("1. Vertical Shear Profile", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Height (m)")
    for i, v in enumerate(speeds):
        ax1.text(v + 0.05, heights[i], f"{v:.2f} m/s", va='center')

    # 2. Wind Probability (Restored from your previous request)
    ax2 = fig.add_subplot(gs[0, 2:4])
    sns.histplot(df['Speed'], kde=True, ax=ax2, color='teal', alpha=0.3, stat='probability')
    ax2.set_title("2. Wind Speed Probability", fontsize=12, fontweight='bold')

    # 3. Directional Energy Rose
    ax3 = fig.add_subplot(gs[0, 4:6], projection='polar')
    bins = np.linspace(0, 2 * np.pi, 36)
    hist, _ = np.histogram(np.deg2rad(df['Direction']), bins=bins, weights=df['WPD'])
    ax3.bar(bins[:-1], hist, width=2 * np.pi / 36, bottom=0.0, color='orange', edgecolor='k', alpha=0.8)
    ax3.set_title("3. Directional Energy Flux", fontsize=12, fontweight='bold', va='bottom')
    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction(-1)

    # --- ROW 2 (EXISTING) ---

    # 4. Tech Comparison
    ax4 = fig.add_subplot(gs[1, 0:3])
    df_sorted = df.sort_values('Timestamp')
    ax4.plot(df_sorted['Timestamp'], df_sorted['Energy_VAWT_kWh'].cumsum(),
             label='VAWT (Baseline)', color='gray', linestyle='-', alpha=0.6)
    ax4.plot(df_sorted['Timestamp'], df_sorted['Energy_HAWT_30m_kWh'].cumsum(),
             label='HAWT (Target)', color='green', linewidth=2.5)
    ax4.set_title("4. Cumulative Energy Yield", fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.3)

    # 5. Regime Timeline (Ghosting)
    ax5 = fig.add_subplot(gs[1, 3:6])
    ax5.plot(df['Timestamp'], df['Speed'], color='lightgray', linewidth=2, zorder=1, label='Wind Flow')
    sns.scatterplot(data=df, x='Timestamp', y='Speed', hue='Regime_Label', palette='deep', s=20, ax=ax5, zorder=2)
    ax5.set_title("5. Regime Stability (Clustering)", fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right')

    # --- ROW 3 (NEW ADDITION: PREDICTABILITY) ---

    ax6 = fig.add_subplot(gs[2, :])  # Full width

    if has_predictions:
        # Calculate Metrics
        valid = df.dropna(subset=['Predicted_Speed'])
        rmse = np.sqrt(mean_squared_error(valid['Speed'], valid['Predicted_Speed']))
        r2 = r2_score(valid['Speed'], valid['Predicted_Speed'])

        # Visualization: Zoom in on the last 150 points for clarity
        subset = valid.tail(150)

        # Plot Actual
        ax6.plot(subset['Timestamp'], subset['Speed'], color='black', alpha=0.3, linewidth=3, label='Actual Speed')
        # Plot Predicted
        ax6.plot(subset['Timestamp'], subset['Predicted_Speed'], color='blue', linestyle='--', marker='d', markersize=4,
                 label='BiLSTM Forecast')

        ax6.set_title(f"6. AI Forecast Accuracy (RMSE: {rmse:.2f} m/s | R²: {r2:.2f})", fontsize=12, fontweight='bold',
                      color='darkblue')
        ax6.legend(loc='upper left')
        ax6.set_ylabel("Wind Speed (m/s)")

        # Add Interpretation Text
        verdict = "HIGH PREDICTABILITY" if r2 > 0.7 else "LOW PREDICTABILITY"
        ax6.text(0.02, 0.05, f"VERDICT: {verdict}", transform=ax6.transAxes, fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8))

    else:
        ax6.text(0.5, 0.5, "Model not trained. Run 'train_offline.py' to see predictions.",
                 ha='center', fontsize=14, color='gray')
        ax6.set_title("6. AI Forecast (Inactive)", fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# --- SUMMARY FUNCTION (PRESERVED) ---
def print_summary(df):
    # Your existing summary logic...
    avg_rho = df['Rho'].mean()
    total_hours = len(df) * 0.25
    rated_cap = 0.45
    total_energy = df['Energy_HAWT_30m_kWh'].sum()
    cf = (total_energy / (rated_cap * total_hours)) * 100

    # Gain Calculation
    gain = ((df['WPD_30m'].mean() - df['WPD'].mean()) / df['WPD'].mean()) * 100

    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print("║          ADVANCED SITING & ENERGY REPORT             ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n1. ENVIRONMENT ANALYSIS")
    print("-" * 30)
    print(f"   • Avg Air Density      : {avg_rho:.3f} kg/m³")
    if avg_rho < 1.15:
        print("     [!] ALERT: Low density (High Temp) is reducing lift.")

    print("\n2. VERTICAL SHEAR ROI")
    print("-" * 30)
    print(f"   • Power Density Gain   : +{gain:.1f}%")

    print("\n3. FEASIBILITY VERDICT")
    print("-" * 30)
    print(f"   • Est. Capacity Factor : {cf:.2f}%")

    # --- NEW ADDITION TO SUMMARY ---
    if 'Predicted_Speed' in df.columns and not df['Predicted_Speed'].isna().all():
        valid = df.dropna(subset=['Predicted_Speed'])
        r2 = r2_score(valid['Speed'], valid['Predicted_Speed'])
        print("\n4. AI PREDICTABILITY")
        print("-" * 30)
        print(f"   • Forecast R-Squared   : {r2:.3f}")
        if r2 > 0.75:
            print("     [✓] Grid Integration : STABLE (Battery requirements low)")
        else:
            print("     [!] Grid Integration : VOLATILE (Battery buffer needed)")

    print("\n" + "=" * 55 + "\n")