# main.py
import data_loader
import physics_engine
import ai_models
import visualizer


def main():
    try:
        # Step 1: Load Data
        print("--- Step 1: Data Ingestion ---")
        raw_data = data_loader.load_and_process_data()

        # Step 2: Apply Physics
        print("\n--- Step 2: Physics Modeling ---")
        phys_data = physics_engine.RefineryPhysics.enrich_data(raw_data)

        # Step 3: AI Analysis
        print("\n--- Step 3: AI Clustering ---")
        ai = ai_models.WindPatternAI()
        final_data = ai.find_regimes(phys_data)

        # Step 4: Report
        print("\n--- Step 4: Generation ---")
        visualizer.print_summary(final_data)
        visualizer.plot_dashboard(final_data)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")


if __name__ == "__main__":
    main()