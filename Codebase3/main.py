import sys
import data_loader
import ai_models
import visualizer


def main():
    try:
        print("╔══════════════════════════════════════════╗")
        print("║   STARTING WIND ASSESSMENT PIPELINE      ║")
        print("╚══════════════════════════════════════════╝")

        # --- Step 1: Data Ingestion ---
        # (Generates the Refinery Data)
        print("\n--- Step 1: Data Ingestion & Preprocessing ---")
        raw_data = data_loader.generate_refinery_data()
        print(f"   -> Loaded {len(raw_data)} timestamps.")
        print("   -> Calculated Vector Components (Wx, Wy).")

        # --- Step 2: Physics & Clustering (The "Old" + "Context") ---
        # We use the WindPatternAI class which now handles:
        # 1. Physics Simulation (Turbine Yields)
        # 2. GMM Clustering (Regime Detection)
        print("\n--- Step 2: Physics Simulation & AI Clustering ---")
        ai_engine = ai_models.WindPatternAI()

        # This single function call now enriches data with:
        # 'Energy_VAWT', 'Energy_HAWT', 'Cluster', 'Regime_Label'
        processed_data = ai_engine.find_regimes(raw_data)
        print("   -> Physics: Calculated Energy Yields for VAWT vs HAWT.")
        print("   -> AI: Identified 3 unique Wind Regimes.")

        # --- Step 3: Reporting & Visualization (The "New") ---
        # The visualizer now automatically:
        # 1. Checks for 'wind_bilstm.pth' (The Trained Brain)
        # 2. Runs the BiLSTM Forecast if the brain exists
        # 3. Plots the Dashboard
        print("\n--- Step 3: Forecasting & Visualization ---")
        visualizer.print_summary(processed_data)
        visualizer.plot_dashboard(processed_data)

    except FileNotFoundError as e:
        print(f"\n[ERROR] Missing File: {e}")
        print("Did you run 'train_offline.py' to generate the model?")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()