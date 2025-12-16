import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics import mean_squared_error, r2_score

import config
from Codebase4 import data_loader
from ai_models import WindBiLSTM


# --- HELPER: LOAD & PREDICT ---
from ai_models import HybridWindModel


def load_and_predict(df):
    if 'Predicted_Speed' in df.columns: return df, True

    try:
        # Load Hybrid
        model = HybridWindModel()
        model.load("lstm_attn.pth", "catboost_resid.cbm")

        with open(config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)

        # Prepare Data (Same logic as training)
        # Note: This is computationally expensive to do every refresh,
        # but ensures accuracy.
        X_seq, X_tab, _, _ = data_loader.prepare_hybrid_tensors(df, config.SEQ_LENGTH)

        # Predict
        preds_scaled = model.predict(X_seq, X_tab)
        preds_actual = scaler.inverse_transform(preds_scaled)

        # Align prediction array with DataFrame
        # Predictions start at index 'seq_len'
        pad_len = config.SEQ_LENGTH
        padding = np.full((pad_len, 1), np.nan)
        full_preds = np.vstack([padding, preds_actual])

        # Ensure length match (trim if necessary due to shifts)
        if len(full_preds) > len(df):
            full_preds = full_preds[:len(df)]
        elif len(full_preds) < len(df):
            # Pad the end if missing
            extra_pad = np.full((len(df) - len(full_preds), 1), np.nan)
            full_preds = np.vstack([full_preds, extra_pad])

        df['Predicted_Speed'] = full_preds
        return df, True

    except Exception as e:
        print(f"Error: {e}")
        df['Predicted_Speed'] = np.nan
        return df, False
# --- PLOTTING HELPERS ---
# These functions accept an 'ax' and a 'fonts' dictionary to control sizing dynamically

def plot_shear(ax, df, f):
    speeds = [df['Speed_10m'].mean(), df['Speed_30m'].mean(), df['Speed_50m'].mean()]
    heights = [10, 30, 50]
    ax.plot(speeds, heights, linestyle='--', color='#95a5a6', alpha=0.6, zorder=1)
    ax.scatter(speeds, heights, color='#e74c3c', s=60, zorder=2, edgecolor='white', linewidth=1)
    ax.set_title("1. Vertical Shear Profile (Hellmann Law)", fontweight='bold', pad=6, fontsize=f['title'])
    ax.set_ylabel("Height (m)", fontsize=f['label'])
    ax.set_xlabel("Mean Wind Speed (m/s)", fontsize=f['label'])
    ax.set_yticks([0, 10, 30, 50, 60])
    ax.tick_params(labelsize=f['tick'])
    for i, v in enumerate(speeds):
        ax.text(v + 0.1, heights[i], f"{v:.2f} m/s", va='center', fontweight='bold', fontsize=f['text'],
                color='#c0392b')


def plot_probability(ax, df, f):
    sns.histplot(df['Speed'], kde=True, ax=ax, color='#1abc9c', alpha=0.4, stat='probability',
                 line_kws={'linewidth': 1})
    ax.set_title("2. Wind Speed Probability", fontweight='bold', pad=6, fontsize=f['title'])
    ax.set_xlabel("Wind Speed (m/s)", fontsize=f['label'])
    ax.set_ylabel("Probability Density", fontsize=f['label'])
    ax.tick_params(labelsize=f['tick'])


def plot_rose(ax, df, f):
    bins = np.linspace(0, 2 * np.pi, 36)
    hist, _ = np.histogram(np.deg2rad(df['Direction']), bins=bins, weights=df['WPD'])
    bars = ax.bar(bins[:-1], hist, width=2 * np.pi / 36, bottom=0.0, edgecolor='k', alpha=0.8)

    import matplotlib.cm as cm
    norm = plt.Normalize(0, max(hist))
    for r, bar in zip(hist, bars):
        bar.set_facecolor(cm.autumn(norm(r)))

    ax.set_title("3. Directional Energy Flux", fontweight='bold', va='bottom', pad=15, fontsize=f['title'])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(labelsize=f['tick'])


def plot_cumulative(ax, df, f):
    df_sorted = df.sort_values('Timestamp')
    ax.plot(df_sorted['Timestamp'], df_sorted['Energy_VAWT_kWh'].cumsum(), label='VAWT (Baseline)', color='#7f8c8d',
            linestyle=':', linewidth=1)
    ax.plot(df_sorted['Timestamp'], df_sorted['Energy_HAWT_30m_kWh'].cumsum(), label='HAWT (Target)', color='#27ae60',
            linewidth=1.5)
    ax.set_title("4. Cumulative Energy Yield (ROI)", fontweight='bold', pad=6, fontsize=f['title'])
    ax.set_ylabel("Total Energy (kWh)", fontsize=f['label'])
    ax.tick_params(labelsize=f['tick'])
    ax.legend(frameon=True, facecolor='white', framealpha=1, loc='upper left', fontsize=f['text'])
    ax.grid(True, linestyle='--', alpha=0.3)


def plot_regimes(ax, df, f):
    ax.plot(df['Timestamp'], df['Speed'], color='#bdc3c7', linewidth=0.8, zorder=1, alpha=0.5, label='Flow')
    sns.scatterplot(data=df, x='Timestamp', y='Speed', hue='Regime_Label', palette='viridis', s=8, ax=ax, zorder=2,
                    edgecolor='none')
    ax.set_title("5. Regime Stability (Clustering)", fontweight='bold', pad=6, fontsize=f['title'])
    ax.tick_params(labelsize=f['tick'])
    ax.legend(loc='upper right', frameon=True, fontsize=f['text'])


def plot_forecast(ax, df, has_predictions, f):
    if has_predictions:
        valid = df.dropna(subset=['Predicted_Speed'])
        if len(valid) > 0:
            rmse = np.sqrt(mean_squared_error(valid['Speed'], valid['Predicted_Speed']))
            r2 = r2_score(valid['Speed'], valid['Predicted_Speed'])
            subset = valid.tail(250)

            ax.plot(subset['Timestamp'], subset['Speed'], color='black', alpha=0.2, linewidth=2, label='Actual Speed')
            ax.plot(subset['Timestamp'], subset['Predicted_Speed'], color='#2980b9', linestyle='-', linewidth=1,
                    marker='o', markersize=2, label='BiLSTM Forecast')

            ax.set_title(f"6. AI Forecast Accuracy (R²: {r2:.3f} | RMSE: {rmse:.2f})", fontweight='bold',
                         color='#2980b9', pad=6, fontsize=f['title'])
            ax.legend(loc='upper left', frameon=True, fontsize=f['text'])
            ax.set_ylabel("Wind Speed (m/s)", fontsize=f['label'])
        else:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center', fontsize=f['title'])
    else:
        ax.text(0.5, 0.5, "Model not trained.", ha='center', fontsize=12, color='gray')
        ax.set_title("6. AI Forecast (Inactive)", fontweight='bold', fontsize=f['title'])
    ax.tick_params(labelsize=f['tick'])


def plot_diurnal(ax, df, f):
    hourly_data = df.groupby(df['Timestamp'].dt.hour)['WPD'].mean()
    hours = hourly_data.index
    values = hourly_data.values
    peak_hour = hourly_data.idxmax()
    peak_val = hourly_data.max()

    colors_hr = ['#3498db' if h != peak_hour else '#e74c3c' for h in hours]
    ax.bar(hours, values, color=colors_hr, alpha=0.85, edgecolor='none', width=0.8)

    ax.set_title("7. Diurnal Energy Profile (Hourly)", fontweight='bold', pad=6, fontsize=f['title'])
    ax.set_xlabel("Hour of Day (00:00 - 23:00)", fontsize=f['label'])
    ax.set_ylabel("Mean Power Density (W/m²)", fontsize=f['label'])
    ax.set_xticks(range(0, 24))
    ax.tick_params(labelsize=f['tick'])
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    ax.annotate(f"GOLDEN HOUR\n{peak_hour}:00",
                xy=(peak_hour, peak_val),
                xytext=(peak_hour + 1, peak_val * 1.05),
                arrowprops=dict(facecolor='#2c3e50', shrink=0.05, width=1),
                fontsize=f['text'], fontweight='bold', color='#2c3e50',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9))

    daily_avg = values.mean()
    ax.axhline(daily_avg, color='#27ae60', linestyle='--', linewidth=1.5, label=f'Avg: {daily_avg:.0f}')
    ax.legend(loc='upper right', fontsize=f['text'])


def plot_seasonal(ax, df, f):
    monthly_data = df.groupby(df['Timestamp'].dt.month)['WPD'].mean()
    monthly_data = monthly_data.reindex(range(1, 13), fill_value=0)
    months = monthly_data.index
    m_values = monthly_data.values
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    peak_month_idx = monthly_data.idxmax()
    peak_month_val = monthly_data.max()
    colors_mo = ['#3498db' if m != peak_month_idx else '#e74c3c' for m in months]

    ax.bar(month_names, m_values, color=colors_mo, alpha=0.85, edgecolor='none')
    ax.set_title("8. Seasonal Energy Profile (Monthly)", fontweight='bold', pad=6, fontsize=f['title'])
    ax.set_ylabel("Mean Power Density (W/m²)", fontsize=f['label'])
    ax.tick_params(labelsize=f['tick'])
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    if peak_month_val > 0:
        best_month_name = month_names[peak_month_idx - 1]
        ax.annotate(f"BEST MONTH\n{best_month_name}",
                    xy=(peak_month_idx - 1, peak_month_val),
                    xytext=(peak_month_idx - 1, peak_month_val * 1.15),
                    arrowprops=dict(facecolor='#2c3e50', shrink=0.05, width=1),
                    fontsize=f['text'], fontweight='bold', color='#2c3e50',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9))


def draw_strategy_card(ax, df, f):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Calculations
    df['Hour'] = df['Timestamp'].dt.hour
    is_peak = (df['Hour'] >= 9) & (df['Hour'] <= 21)
    peak_speed = df[is_peak]['Speed'].mean()
    off_speed = df[~is_peak]['Speed'].mean()
    revenue_status = "[$$$] PRIME SITE" if peak_speed > off_speed else "[$] NIGHT OWL"
    revenue_desc = "Daytime Peak" if peak_speed > off_speed else "Off-Peak Skew"

    dir_spread = df['Direction'].std()
    land_status = "[✓] UNI-DIRECTIONAL" if dir_spread < 30 else "[~] BI-DIRECTIONAL" if dir_spread < 60 else "[!] OMNI-DIRECTIONAL"

    df['Is_Calm'] = df['Speed'] < 3.0
    df['Group_ID'] = (df['Is_Calm'] != df['Is_Calm'].shift()).cumsum()
    calm_periods = df[df['Is_Calm']].groupby('Group_ID').size()
    max_drought = calm_periods.max() * 0.25 if not calm_periods.empty else 0
    rel_status = "[!] HIGH RISK" if max_drought > 24 else "[✓] RELIABLE"

    df['Delta_WPD'] = df['WPD'].diff()
    max_drop = df['Delta_WPD'].min()
    grid_status = "[!] GRID SHOCK" if max_drop < -100 else "[✓] GRID STABLE"

    avg_turb = df['Speed'].rolling(4).std().mean()
    turb_status = "[!] HIGH STRESS" if avg_turb > 1.5 else "[✓] SMOOTH FLOW"

    # Drawing Text
    ax.text(0.5, 0.95, "9. STRATEGIC & OPERATIONAL INSIGHTS SCORECARD",
            ha='center', va='top', fontsize=f['title'], fontweight='bold', color='#2c3e50',
            bbox=dict(facecolor='#ecf0f1', edgecolor='#bdc3c7', pad=4))

    # Left Column
    ax.text(0.05, 0.82, "1. REVENUE PROFILE (Value Factor)", fontsize=f['label'], fontweight='bold', color='#34495e')
    ax.text(0.05, 0.76, f"• Peak (Day): {peak_speed:.2f} m/s | Off-Peak: {off_speed:.2f} m/s", fontsize=f['text'])
    ax.text(0.05, 0.70, f"{revenue_status}: {revenue_desc}", fontsize=f['text'], fontweight='bold',
            color='#27ae60' if '>' in revenue_status else '#e67e22')

    ax.text(0.05, 0.45, "2. LAND USE EFFICIENCY (Wake Risk)", fontsize=f['label'], fontweight='bold', color='#34495e')
    ax.text(0.05, 0.39, f"• Direction Spread: {dir_spread:.1f}°", fontsize=f['text'])
    ax.text(0.05, 0.33, f"{land_status}", fontsize=f['text'], fontweight='bold',
            color='#c0392b' if '!' in land_status else '#27ae60')

    # Right Column
    ax.text(0.55, 0.82, "3. RELIABILITY (Wind Drought)", fontsize=f['label'], fontweight='bold', color='#34495e')
    ax.text(0.55, 0.76, f"• Max Zero-Gen Duration: {max_drought:.1f} hours", fontsize=f['text'])
    ax.text(0.55, 0.70, f"{rel_status}", fontsize=f['text'], fontweight='bold',
            color='#c0392b' if '!' in rel_status else '#27ae60')

    ax.text(0.55, 0.45, "4. OPERATIONAL SAFETY", fontsize=f['label'], fontweight='bold', color='#34495e')
    ax.text(0.55, 0.39, f"{grid_status}: Max Drop {max_drop:.0f} W/m²", fontsize=f['text'], fontweight='bold',
            color='#c0392b' if '!' in grid_status else '#27ae60')
    ax.text(0.55, 0.33, f"{turb_status}: Turbulence {avg_turb:.2f} m/s", fontsize=f['text'], fontweight='bold',
            color='#c0392b' if '!' in turb_status else '#27ae60')

    rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='#7f8c8d', fill=False, linewidth=1)
    ax.add_patch(rect)
    ax.plot([0.05, 0.95], [0.55, 0.55], color='#bdc3c7', linewidth=0.5, linestyle='--')


# --- MAIN PLOTTING ENGINE ---
def plot_dashboard(df):
    # 1. Prepare Data
    df, has_predictions = load_and_predict(df)
    print("Generating Advanced Dashboard and Individual Plots...")

    plt.style.use('seaborn-v0_8-whitegrid')

    # --- DEFINE FONT PROFILES ---
    # Profile A: For the Combined Dashboard (Extremely Compact)
    dash_fonts = {
        'title': 9,
        'label': 7,
        'text': 6,
        'tick': 6
    }

    # Profile B: For Individual Windows (Normal/Readable)
    indiv_fonts = {
        'title': 14,
        'label': 12,
        'text': 10,
        'tick': 10
    }

    # ---------------------------------------------
    # 1. MAIN DASHBOARD (COMBINED)
    # ---------------------------------------------
    # Use context manager for dashboard specific rcParams
    with plt.rc_context({'font.size': 6, 'figure.titlesize': 16}):
        fig = plt.figure(figsize=(22, 45))
        fig.canvas.manager.set_window_title("Master Dashboard")

        gs = gridspec.GridSpec(6, 6, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1.8], hspace=0.8, wspace=0.3)
        fig.suptitle('Advanced Wind Energy Site Assessment', fontweight='bold', y=0.985, color='#2c3e50')

        # Calls with 'dash_fonts'
        plot_shear(fig.add_subplot(gs[0, 0:2]), df, dash_fonts)
        plot_probability(fig.add_subplot(gs[0, 2:4]), df, dash_fonts)
        plot_rose(fig.add_subplot(gs[0, 4:6], projection='polar'), df, dash_fonts)
        plot_cumulative(fig.add_subplot(gs[1, 0:3]), df, dash_fonts)
        plot_regimes(fig.add_subplot(gs[1, 3:6]), df, dash_fonts)
        plot_forecast(fig.add_subplot(gs[2, :]), df, has_predictions, dash_fonts)
        plot_diurnal(fig.add_subplot(gs[3, :]), df, dash_fonts)
        plot_seasonal(fig.add_subplot(gs[4, :]), df, dash_fonts)
        draw_strategy_card(fig.add_subplot(gs[5, :]), df, dash_fonts)

        plt.tight_layout(rect=[0, 0, 1, 0.97])

    # ---------------------------------------------
    # 2. INDIVIDUAL WINDOWS
    # ---------------------------------------------
    # Use context manager for individual plots to have normal sizing
    with plt.rc_context({'font.size': 10, 'figure.titlesize': 14}):
        # Plot 1: Shear
        fig1 = plt.figure(figsize=(10, 8))
        fig1.canvas.manager.set_window_title("1. Vertical Shear")
        plot_shear(fig1.add_subplot(111), df, indiv_fonts)
        plt.tight_layout()

        # Plot 2: Probability
        fig2 = plt.figure(figsize=(10, 8))
        fig2.canvas.manager.set_window_title("2. Probability Distribution")
        plot_probability(fig2.add_subplot(111), df, indiv_fonts)
        plt.tight_layout()

        # Plot 3: Rose
        fig3 = plt.figure(figsize=(10, 10))
        fig3.canvas.manager.set_window_title("3. Wind Rose")
        plot_rose(fig3.add_subplot(111, projection='polar'), df, indiv_fonts)
        plt.tight_layout()

        # Plot 4: Cumulative
        fig4 = plt.figure(figsize=(12, 8))
        fig4.canvas.manager.set_window_title("4. Cumulative Yield")
        plot_cumulative(fig4.add_subplot(111), df, indiv_fonts)
        plt.tight_layout()

        # Plot 5: Regimes
        fig5 = plt.figure(figsize=(12, 8))
        fig5.canvas.manager.set_window_title("5. Regime Stability")
        plot_regimes(fig5.add_subplot(111), df, indiv_fonts)
        plt.tight_layout()

        # Plot 6: Forecast
        fig6 = plt.figure(figsize=(12, 8))
        fig6.canvas.manager.set_window_title("6. AI Forecast")
        plot_forecast(fig6.add_subplot(111), df, has_predictions, indiv_fonts)
        plt.tight_layout()

        # Plot 7: Diurnal
        fig7 = plt.figure(figsize=(12, 8))
        fig7.canvas.manager.set_window_title("7. Diurnal Profile")
        plot_diurnal(fig7.add_subplot(111), df, indiv_fonts)
        plt.tight_layout()

        # Plot 8: Seasonal
        fig8 = plt.figure(figsize=(12, 8))
        fig8.canvas.manager.set_window_title("8. Seasonal Profile")
        plot_seasonal(fig8.add_subplot(111), df, indiv_fonts)
        plt.tight_layout()

        # Plot 9: Strategy Card (Adjusted size for readability in standalone)
        fig9 = plt.figure(figsize=(12, 8))
        fig9.canvas.manager.set_window_title("9. Strategy Scorecard")
        draw_strategy_card(fig9.add_subplot(111), df, indiv_fonts)
        plt.tight_layout()

    # Show all windows at once
    plt.show()


def print_summary(df):
    avg_rho = df['Rho'].mean()
    print("\n" + "=" * 60)
    print("      PHYSICAL ENVIRONMENT REPORT")
    print("=" * 60)
    print(f"1. AIR PROPERTIES")
    print(f"   • Avg Density        : {avg_rho:.3f} kg/m³")
    if avg_rho < 1.15: print("     [!] Low Density: High temperature derating likely.")
    print("\n[INFO] Dashboard and Individual Plots Launched.")