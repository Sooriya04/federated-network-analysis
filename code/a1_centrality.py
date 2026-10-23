"""
Assignment I - Task 2: Centrality Measures and Correlation Analysis
- Computes In-Degree, Out-Degree, Closeness, Betweenness, and Eigenvector centralities.
- Supports exact calculation or k-sampling approximation for betweenness with runtime reporting.
- Exports Top-10 tables for each centrality measure as CSV.
- Computes Pearson and Spearman correlation matrices across measures and exports as CSV.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ACTIVE_DATASET,
    OUTPUTS_DIR,
    RANDOM_SEED
)
from src.a1_preprocess import load_and_preprocess

def compute_centralities(dataset_key: str = ACTIVE_DATASET, betweenness_k: int = None):
    """
    Computes degree (in/out), closeness, betweenness, and eigenvector centralities.
    Uses full cleaned graph for degrees and eigenvector; uses largest WCC for closeness/betweenness
    to guarantee finite shortest paths.
    """
    print(f"\n============================================================")
    print(f"TASK 2: CENTRALITY ANALYSIS ({dataset_key})")
    print(f"============================================================")

    G, G_wcc, G_scc, _ = load_and_preprocess(dataset_key)
    n_total = G.number_of_nodes()
    n_wcc = G_wcc.number_of_nodes()

    results = {}
    runtimes = {}

    # 1. Degree Centrality (In and Out on full directed graph)
    t0 = time.time()
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    # Normalized by (N - 1)
    norm = n_total - 1 if n_total > 1 else 1
    results["in_degree"] = in_deg
    results["out_degree"] = out_deg
    results["in_degree_norm"] = {k: v / norm for k, v in in_deg.items()}
    results["out_degree_norm"] = {k: v / norm for k, v in out_deg.items()}
    runtimes["degree"] = time.time() - t0
    print(f"[Done] In/Out Degree Centrality calculated in {runtimes['degree']:.3f}s")

    # 2. Closeness Centrality
    # Evaluated on largest WCC (reversed edges for incoming closeness / reachability to node)
    t0 = time.time()
    closeness = nx.closeness_centrality(G_wcc)
    # Map back to full node set (nodes outside WCC receive 0.0)
    results["closeness"] = {node: closeness.get(node, 0.0) for node in G.nodes()}
    runtimes["closeness"] = time.time() - t0
    print(f"[Done] Closeness Centrality calculated in {runtimes['closeness']:.3f}s on WCC ({n_wcc} nodes)")

    # 3. Betweenness Centrality
    t0 = time.time()
    if betweenness_k is not None and betweenness_k < n_wcc:
        print(f"Computing sampled Betweenness Centrality (k={betweenness_k}, seed={RANDOM_SEED})...")
        betweenness = nx.betweenness_centrality(
            G_wcc, k=betweenness_k, normalized=True, seed=RANDOM_SEED
        )
    else:
        print(f"Computing exact Betweenness Centrality across all {n_wcc} nodes...")
        betweenness = nx.betweenness_centrality(G_wcc, normalized=True)
    results["betweenness"] = {node: betweenness.get(node, 0.0) for node in G.nodes()}
    runtimes["betweenness"] = time.time() - t0
    print(f"[Done] Betweenness Centrality calculated in {runtimes['betweenness']:.3f}s (k={'exact' if betweenness_k is None else betweenness_k})")

    # 4. Eigenvector Centrality
    t0 = time.time()
    try:
        # For directed graphs, standard eigenvector centrality uses left/in-degree authority
        eigenvector = nx.eigenvector_centrality(G, max_iter=2000, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        print("Standard power iteration failed; falling back to numpy eigenvector on largest WCC...")
        eigenvector = nx.eigenvector_centrality_numpy(G_wcc)
    results["eigenvector"] = {node: eigenvector.get(node, 0.0) for node in G.nodes()}
    runtimes["eigenvector"] = time.time() - t0
    print(f"[Done] Eigenvector Centrality calculated in {runtimes['eigenvector']:.3f}s")

    # 5. Compile Centrality DataFrame
    df_centrality = pd.DataFrame({
        "node": list(G.nodes()),
        "in_degree": [results["in_degree"][n] for n in G.nodes()],
        "out_degree": [results["out_degree"][n] for n in G.nodes()],
        "in_degree_norm": [results["in_degree_norm"][n] for n in G.nodes()],
        "out_degree_norm": [results["out_degree_norm"][n] for n in G.nodes()],
        "closeness": [results["closeness"][n] for n in G.nodes()],
        "betweenness": [results["betweenness"][n] for n in G.nodes()],
        "eigenvector": [results["eigenvector"][n] for n in G.nodes()]
    })

    # Save full centrality table
    full_csv = OUTPUTS_DIR / f"{dataset_key}_centrality_all.csv"
    df_centrality.to_csv(full_csv, index=False)
    print(f"Saved complete centralities to: {full_csv}")

    # 6. Extract Top-10 Tables for Each Measure
    measures = [
        ("in_degree", "In-Degree (Nominations)"),
        ("out_degree", "Out-Degree (Following)"),
        ("closeness", "Closeness Centrality"),
        ("betweenness", "Betweenness Centrality"),
        ("eigenvector", "Eigenvector Centrality")
    ]

    top10_records = []
    for col, display_name in measures:
        top10 = df_centrality.sort_values(by=col, ascending=False).head(10).reset_index(drop=True)
        top10_path = OUTPUTS_DIR / f"{dataset_key}_top10_{col}.csv"
        top10[["node", col]].rename(columns={"node": "Node ID", col: "Score"}).to_csv(top10_path, index=False)
        print(f"\nTop-10 by {display_name}:")
        for rank, row in top10.iterrows():
            val = row[col]
            val_str = f"{int(val)}" if col.endswith("_degree") else f"{val:.6f}"
            print(f"  Rank {rank+1:2d}: {row['node']:<35s} | {val_str}")
            top10_records.append({
                "Measure": display_name,
                "Rank": rank + 1,
                "Node ID": row["node"],
                "Score": val
            })

    # Combined Top-10 summary table
    df_top10_all = pd.DataFrame(top10_records)
    df_top10_all.to_csv(OUTPUTS_DIR / f"{dataset_key}_top10_centralities_combined.csv", index=False)

    # 7. Correlation Matrices (Pearson and Spearman)
    cols_to_correlate = ["in_degree", "out_degree", "closeness", "betweenness", "eigenvector"]
    pearson_corr = df_centrality[cols_to_correlate].corr(method="pearson").round(4)
    spearman_corr = df_centrality[cols_to_correlate].corr(method="spearman").round(4)

    pearson_path = OUTPUTS_DIR / f"{dataset_key}_centrality_pearson_corr.csv"
    spearman_path = OUTPUTS_DIR / f"{dataset_key}_centrality_spearman_corr.csv"
    pearson_corr.to_csv(pearson_path)
    spearman_corr.to_csv(spearman_path)

    print(f"\nPearson Correlation Matrix:\n{pearson_corr}")
    print(f"\nSpearman Rank Correlation Matrix:\n{spearman_corr}")

    # Summary of runtimes and parameters
    run_meta = pd.DataFrame({
        "Measure": list(runtimes.keys()),
        "Runtime_Seconds": [round(v, 4) for v in runtimes.values()],
        "Sample_Size_k": ["N/A", "N/A", "exact" if betweenness_k is None else betweenness_k, "N/A"]
    })
    run_meta.to_csv(OUTPUTS_DIR / f"{dataset_key}_centrality_runtimes.csv", index=False)

    return df_centrality, pearson_corr, spearman_corr

if __name__ == "__main__":
    compute_centralities()
