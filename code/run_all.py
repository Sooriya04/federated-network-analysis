"""
Unified Execution Script for Social Network Analysis Project (22ITPA0)
Runs all Assignment I and Assignment II pipelines end-to-end:
  1. Data Preprocessing & Cleaning (a1_preprocess.py)
  2. Centrality Analysis (a1_centrality.py)
  3. Prestige Measures (a1_prestige.py)
  4. PageRank & HITS Algorithms (a1_pagerank_hits.py)
  5. Preferential Attachment & Power-Law (a1_pref_attachment.py)
  6. Small-World Topology Analysis (a1_small_world.py)
  7. Visualizations & Gephi GEXF Export (viz.py)
  8. Link Prediction Benchmark (a2_link_prediction.py)
  9. Granovetter Threshold Cascades (a2_granovetter.py)
  10. Random Walk & Convergence Analysis (a2_random_walk.py)
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ACTIVE_DATASET, OUTPUTS_DIR, DATASET_CONFIGS
from src.a1_preprocess import load_and_preprocess
from src.a1_centrality import compute_centralities
from src.a1_prestige import compute_prestige
from src.a1_pagerank_hits import compute_pagerank_and_hits
from src.a1_pref_attachment import analyze_preferential_attachment
from src.a1_small_world import compute_small_world
from src.viz import generate_all_visualizations
from src.a2_link_prediction import run_link_prediction
from src.a2_granovetter import simulate_real_network_cascades
from src.a2_random_walk import run_random_walk_analysis

def main():
    parser = argparse.ArgumentParser(description="Run complete Social Network Analysis pipeline.")
    parser.add_argument("--dataset", type=str, default=ACTIVE_DATASET, choices=["peertube", "wiki_vote"],
                        help="Dataset key to analyze ('peertube' or 'wiki_vote').")
    parser.add_argument("--betweenness_k", type=int, default=None,
                        help="Optional sample size k for betweenness centrality (default: exact).")
    args = parser.parse_args()

    dataset_key = args.dataset
    cfg = DATASET_CONFIGS[dataset_key]

    print("\n" + "="*70)
    print("SOCIAL NETWORK ANALYSIS (22ITPA0) - COMPLETE PIPELINE")
    print(f"Active Dataset: {cfg['name']}")
    print(f"Citation:       {cfg['citation']}")
    print(f"Outputs Folder: {OUTPUTS_DIR}")
    print("="*70)

    total_start = time.time()
    timings = {}

    pipeline = [
        ("Step 1: Data Preprocessing & Cleaning", lambda: load_and_preprocess(dataset_key)),
        ("Step 2: Centrality Measures & Correlations", lambda: compute_centralities(dataset_key, args.betweenness_k)),
        ("Step 3: Prestige Measures (In-Degree, Proximity, Katz)", lambda: compute_prestige(dataset_key)),
        ("Step 4 & 5: PageRank & HITS Algorithms", lambda: compute_pagerank_and_hits(dataset_key)),
        ("Step 6: Preferential Attachment & Power-Law Fit", lambda: analyze_preferential_attachment(dataset_key)),
        ("Step 7: Small-World Invariant Analysis", lambda: compute_small_world(dataset_key)),
        ("Step 8: Visualizations & Gephi GEXF Export", lambda: generate_all_visualizations(dataset_key)),
        ("Step 9 (A2.1): Link Prediction Heuristics & AUC/Precision", lambda: run_link_prediction(dataset_key)),
        ("Step 10 (A2.2): Granovetter Threshold Diffusion", lambda: simulate_real_network_cascades(dataset_key)),
        ("Step 11 (A2.3): Random Walk & Stationary Convergence", lambda: run_random_walk_analysis(dataset_key)),
    ]

    for step_name, func in pipeline:
        print("\n" + "#"*70)
        print(f"RUNNING: {step_name}")
        print("#"*70)
        t_step = time.time()
        func()
        elapsed = time.time() - t_step
        timings[step_name] = elapsed
        print(f"--> Completed in {elapsed:.2f}s")

    total_elapsed = time.time() - total_start
    print("\n" + "="*70)
    print(f"ALL TASKS COMPLETED SUCCESSFULLY IN {total_elapsed:.2f}s ({total_elapsed/60:.2f} mins)")
    print("="*70)
    print("\nStage Runtime Breakdown:")
    for step_name, elapsed in timings.items():
        print(f"  {step_name:<50s} : {elapsed:6.2f}s")
    print(f"\nAll tables (CSV), figures (PNG 300 dpi), and Gephi files (GEXF) are saved in:\n  {OUTPUTS_DIR}")

if __name__ == "__main__":
    main()
