# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_dashboard(df):
    print("Generating Visualizations...")
    plt.figure(figsize=(16, 10))

    # 1. Wind Rose (Direction Distribution)
    ax = plt.subplot(2, 2, 1, projection='polar')
    ax.set_title("Wind Direction Distribution", va='bottom', fontweight='bold')

    # Create histogram bins
    bins = np.linspace(0, 2 * np.pi, 36)
    hist, _ = np.histogram(np.deg2rad(df['Direction']), bins=bins)

    # Plot bars
    ax.bar(bins[:-1], hist, width=2 * np.pi / 36, bottom=0.0, color='teal', alpha=0.7, edgecolor='black')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # Clockwise

    # 2. Power vs Heat (Refinery Specific)
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='Temp', y='WPD', hue='Regime_Label', palette='viridis', alpha=0.8)
    plt.title("Turbine Output Efficiency vs Temperature")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Wind Power Density (W/m²)")
    plt.grid(True, linestyle='--', alpha=0.3)

    # 3. Regime Timeline
    plt.subplot(2, 1, 2)
    sns.scatterplot(data=df, x='Timestamp', y='Speed', hue='Regime_Label', palette='deep', s=20)
    plt.title("Identified Wind Regimes Over Time")
    plt.ylabel("Wind Speed (m/s)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def print_summary(df):
    avg_wpd = df['WPD'].mean()
    avg_rho = df['Rho'].mean()

    print("\n" + "=" * 40)
    print("      FINAL SITING ASSESSMENT")
    print("=" * 40)
    print(f"Average Air Density: {avg_rho:.3f} kg/m3")
    print("   (Note: Standard Sea Level density is 1.225 kg/m3)")

    if avg_rho < 1.15:
        print(">> WARNING: Low Air Density detected (Refinery Heat Island).")
        print(">> ACTION: Turbines will underperform. Use larger rotor blades.")

    print(f"\nAverage Power Density: {avg_wpd:.2f} W/m2")

    if avg_wpd > 250:
        print(">> VERDICT: EXCELLENT Site for Mini Turbines.")
    elif avg_wpd > 150:
        print(">> VERDICT: GOOD. Viable for high-efficiency models.")
    else:
        print(">> VERDICT: POOR. Wind resource is likely insufficient.")
    print("=" * 40 + "\n")