# # visualizer.py
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
#
# def plot_dashboard(df):
#     print("Generating Visualizations...")
#     plt.figure(figsize=(16, 10))
#
#     # 1. Wind Rose (Direction Distribution)
#     ax = plt.subplot(2, 2, 1, projection='polar')
#     ax.set_title("Wind Direction Distribution", va='bottom', fontweight='bold')
#
#     # Create histogram bins
#     bins = np.linspace(0, 2 * np.pi, 36)
#     hist, _ = np.histogram(np.deg2rad(df['Direction']), bins=bins)
#
#     # Plot bars
#     ax.bar(bins[:-1], hist, width=2 * np.pi / 36, bottom=0.0, color='teal', alpha=0.7, edgecolor='black')
#     ax.set_theta_zero_location("N")
#     ax.set_theta_direction(-1)  # Clockwise
#
#     # 2. Power vs Heat (Refinery Specific)
#     plt.subplot(2, 2, 2)
#     sns.scatterplot(data=df, x='Temp', y='WPD', hue='Regime_Label', palette='viridis', alpha=0.8)
#     plt.title("Turbine Output Efficiency vs Temperature")
#     plt.xlabel("Temperature (°C)")
#     plt.ylabel("Wind Power Density (W/m²)")
#     plt.grid(True, linestyle='--', alpha=0.3)
#
#     # 3. Regime Timeline
#     plt.subplot(2, 1, 2)
#     sns.scatterplot(data=df, x='Timestamp', y='Speed', hue='Regime_Label', palette='deep', s=20)
#     plt.title("Identified Wind Regimes Over Time")
#     plt.ylabel("Wind Speed (m/s)")
#     plt.grid(True)
#
#     plt.tight_layout()
#     plt.show()
#
#
# def print_summary(df):
#     avg_wpd = df['WPD'].mean()
#     avg_rho = df['Rho'].mean()
#
#     print("\n" + "=" * 40)
#     print("      FINAL SITING ASSESSMENT")
#     print("=" * 40)
#     print(f"Average Air Density: {avg_rho:.3f} kg/m3")
#     print("   (Note: Standard Sea Level density is 1.225 kg/m3)")
#
#     if avg_rho < 1.15:
#         print(">> WARNING: Low Air Density detected (Refinery Heat Island).")
#         print(">> ACTION: Turbines will underperform. Use larger rotor blades.")
#
#     print(f"\nAverage Power Density: {avg_wpd:.2f} W/m2")
#
#     if avg_wpd > 250:
#         print(">> VERDICT: EXCELLENT Site for Mini Turbines.")
#     elif avg_wpd > 150:
#         print(">> VERDICT: GOOD. Viable for high-efficiency models.")
#     else:
#         print(">> VERDICT: POOR. Wind resource is likely insufficient.")
#     print("=" * 40 + "\n")

# visualizer.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np


def plot_dashboard(df):
    print("Generating Advanced Dashboard...")

    # UI: Use a clean style
    plt.style.use('seaborn-v0_8-whitegrid')

    # UI: Set up a Grid Layout (2 Rows, 3 Columns)
    # Top Row: 3 smaller summary charts
    # Bottom Row: 2 wider timeline charts
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 6, figure=fig)

    # Main Title
    fig.suptitle('Advanced Wind Energy Site Assessment', fontsize=20, fontweight='bold', y=0.95)

    # --- 1. Vertical Profile (Top Left) ---
    ax1 = fig.add_subplot(gs[0, 0:2])  # Span 2 columns
    speeds = [df['Speed_10m'].mean(), df['Speed_30m'].mean(), df['Speed_50m'].mean()]
    heights = [10, 30, 50]

    # Visual: "Old" Context (Connector) vs "New" Data (Dots)
    ax1.plot(speeds, heights, linestyle='--', color='gray', alpha=0.5)
    ax1.plot(speeds, heights, marker='o', linewidth=0, color='crimson', markersize=10)

    ax1.set_title("1. Vertical Shear Profile", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Mean Wind Speed (m/s)")
    ax1.set_ylabel("Height (m)")

    # Annotate points clearly
    for i, v in enumerate(speeds):
        ax1.text(v + 0.02, heights[i], f" {v:.2f} m/s", va='center', fontweight='bold', fontsize=9)

    # --- 2. Wind Probability Distribution (Top Middle - RESTORED) ---
    ax2 = fig.add_subplot(gs[0, 2:4])  # Span 2 columns

    # Visual: Histogram (Bars) = Raw Data | KDE (Line) = Probability Trend
    sns.histplot(df['Speed'], kde=True, ax=ax2, color='teal', alpha=0.3, stat='probability')

    ax2.set_title("2. Wind Speed Probability", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Wind Speed (m/s)")
    ax2.set_ylabel("Probability")

    # Mark the average
    avg_speed = df['Speed'].mean()
    ax2.axvline(avg_speed, color='red', linestyle='--', label=f'Avg: {avg_speed:.1f} m/s')
    ax2.legend()

    # --- 3. Directional Energy Rose (Top Right) ---
    ax3 = fig.add_subplot(gs[0, 4:6], projection='polar')  # Span 2 columns
    bins = np.linspace(0, 2 * np.pi, 36)
    hist, _ = np.histogram(np.deg2rad(df['Direction']), bins=bins, weights=df['WPD'])

    ax3.bar(bins[:-1], hist, width=2 * np.pi / 36, bottom=0.0, color='orange', edgecolor='k', alpha=0.8)
    ax3.set_title("3. Directional Energy Flux", fontsize=12, fontweight='bold', va='bottom')
    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction(-1)

    # --- 4. Tech Comparison (Bottom Left - Wide) ---
    ax4 = fig.add_subplot(gs[1, 0:3])  # Span 3 columns (Half width)
    df_sorted = df.sort_values('Timestamp')

    # Visual Hierarchy: Ghost out the baseline, Highlight the target
    ax4.plot(df_sorted['Timestamp'], df_sorted['Energy_VAWT_kWh'].cumsum(),
             label='VAWT (Baseline)', color='gray', linestyle='-', alpha=0.5)

    ax4.plot(df_sorted['Timestamp'], df_sorted['Energy_HAWT_30m_kWh'].cumsum(),
             label='HAWT 30m (Target)', color='green', linewidth=2.5)

    ax4.set_title("4. Cumulative Energy Yield", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Energy (kWh)")
    ax4.legend(loc='upper left', frameon=True)
    ax4.grid(True, linestyle='--', alpha=0.3)

    # --- 5. Regime Timeline (Bottom Right - Wide) ---
    ax5 = fig.add_subplot(gs[1, 3:6])  # Span 3 columns (Half width)

    # GHOST LAYER (Context): Continuous gray line for flow
    ax5.plot(df_sorted['Timestamp'], df_sorted['Speed'], color='lightgray', linewidth=2, zorder=1, label='Wind Flow')

    # FOCUS LAYER (Events): Colored dots for regimes
    sns.scatterplot(data=df, x='Timestamp', y='Speed', hue='Regime_Label', palette='deep', s=30, ax=ax5, zorder=2)

    ax5.set_title("5. Regime Stability (Ghosting Technique)", fontsize=12, fontweight='bold')
    ax5.set_ylabel("Wind Speed (m/s)")
    ax5.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax5.grid(True, alpha=0.3)

    # Final Spacing Adjustment
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave room for the main title
    plt.show()


def print_summary(df):
    avg_wpd = df['WPD'].mean()
    avg_rho = df['Rho'].mean()

    # Calculations
    total_hours = len(df) * 0.25
    rated_cap = 0.45
    total_energy = df['Energy_HAWT_30m_kWh'].sum()
    cf = (total_energy / (rated_cap * total_hours)) * 100

    # Gain Calculation
    gain = ((df['WPD_30m'].mean() - df['WPD'].mean()) / df['WPD'].mean()) * 100

    # Formatted Output
    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print("║          ADVANCED SITING & ENERGY REPORT             ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n1. ENVIRONMENT ANALYSIS")
    print("-" * 30)
    print(f"   • Avg Air Density      : {avg_rho:.3f} kg/m³")
    print(f"   • Std Air Density      : 1.225 kg/m³")
    if avg_rho < 1.15:
        print("     [!] ALERT: Low density (High Temp) is reducing lift.")

    print("\n2. VERTICAL SHEAR ROI")
    print("-" * 30)
    print(f"   • 10m Speed (Current)  : {df['Speed_10m'].mean():.2f} m/s")
    print(f"   • 30m Speed (Target)   : {df['Speed_30m'].mean():.2f} m/s")
    print(f"   • Power Density Gain   : +{gain:.1f}%")
    print("     [i] Insight: Building higher yields significantly more power.")

    print("\n3. FEASIBILITY VERDICT")
    print("-" * 30)
    print(f"   • Est. Capacity Factor : {cf:.2f}% (HAWT Class)")

    print("\n   >>> FINAL DECISION:")
    if cf > 20:
        print("       ★ EXCELLENT SITE. Proceed with standard HAWT turbines.")
    elif cf > 10:
        print("       ⚠️ MARGINAL SITE. Consider VAWTs or low-speed optimized rotors.")
    else:
        print("       ⛔ POOR SITE. Wind resource insufficient. Recommend Solar PV.")

    print("\n" + "=" * 55 + "\n")