"""
Assignment I - Task 4 & 5: PageRank and HITS Algorithms
- PageRank with damping factor 0.85, 0.50, and 0.99.
- Spearman rank correlation comparison between PageRank and In-Degree.
- Quantitative rank shift analysis across damping values.
- HITS algorithm: Top-10 Hubs and Top-10 Authorities.
- Analysis of structural asymmetry: nodes high in Hub score but low in Authority, and vice-versa.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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

def compute_pagerank_and_hits(dataset_key: str = ACTIVE_DATASET):
    """
    Computes PageRank under varying damping factors and HITS hub/authority scores.
    Exports top-10 tables, rank shift matrices, and structural asymmetry analyses.
    """
    print(f"\n============================================================")
    print(f"TASKS 4 & 5: PAGERANK AND HITS ANALYSIS ({dataset_key})")
    print(f"============================================================")

    G, G_wcc, _, _ = load_and_preprocess(dataset_key)
    nodes = list(G.nodes())
    in_degrees = dict(G.in_degree())

    # ============================================================
    # TASK 4: PAGERANK ANALYSIS
    # ============================================================
    print("\n--- Running PageRank (damping = 0.85, 0.50, 0.99) ---")
    damping_factors = [0.85, 0.50, 0.99]
    pr_scores = {}
    pr_runtimes = {}

    for d in damping_factors:
        t0 = time.time()
        pr = nx.pagerank(G, alpha=d, max_iter=2000, tol=1e-07)
        pr_scores[d] = pr
        pr_runtimes[d] = time.time() - t0
        print(f"  PageRank (alpha={d:.2f}) converged in {pr_runtimes[d]:.3f}s")

    # PageRank DataFrame
    df_pr = pd.DataFrame({
        "node": nodes,
        "in_degree": [in_degrees[n] for n in nodes],
        "pagerank_085": [pr_scores[0.85][n] for n in nodes],
        "pagerank_050": [pr_scores[0.50][n] for n in nodes],
        "pagerank_099": [pr_scores[0.99][n] for n in nodes]
    })

    # Add Ranks (1 = highest score)
    df_pr["rank_in_degree"] = df_pr["in_degree"].rank(ascending=False, method="min").astype(int)
    df_pr["rank_pr_085"] = df_pr["pagerank_085"].rank(ascending=False, method="min").astype(int)
    df_pr["rank_pr_050"] = df_pr["pagerank_050"].rank(ascending=False, method="min").astype(int)
    df_pr["rank_pr_099"] = df_pr["pagerank_099"].rank(ascending=False, method="min").astype(int)

    # Save full PageRank table
    pr_full_path = OUTPUTS_DIR / f"{dataset_key}_pagerank_all.csv"
    df_pr.to_csv(pr_full_path, index=False)

    # Extract Top-10 by PR(0.85)
    top10_pr085 = df_pr.sort_values(by="pagerank_085", ascending=False).head(10).reset_index(drop=True)
    top10_pr085_path = OUTPUTS_DIR / f"{dataset_key}_top10_pagerank_085.csv"
    top10_pr085[["node", "pagerank_085", "in_degree", "rank_in_degree"]].rename(
        columns={"node": "Node ID", "pagerank_085": "PageRank (d=0.85)", "in_degree": "In-Degree", "rank_in_degree": "In-Degree Rank"}
    ).to_csv(top10_pr085_path, index=False)

    print(f"\nTop-10 by PageRank (d=0.85):")
    for idx, row in top10_pr085.iterrows():
        print(f"  Rank {idx+1:2d}: {row['node']:<35s} | PR: {row['pagerank_085']:.6f} | In-Deg: {int(row['in_degree']):3d} (Rank {int(row['rank_in_degree']):2d})")

    # Spearman Correlation: PageRank (0.85) vs In-Degree
    spearman_pr_indeg, p_val_indeg = spearmanr(df_pr["pagerank_085"], df_pr["in_degree"])
    print(f"\nSpearman Correlation between PageRank (d=0.85) and In-Degree:")
    print(f"  rho = {spearman_pr_indeg:.4f} (p-value = {p_val_indeg:.2e})")

    # Damping Shift Analysis
    rho_050_085, _ = spearmanr(df_pr["pagerank_050"], df_pr["pagerank_085"])
    rho_085_099, _ = spearmanr(df_pr["pagerank_085"], df_pr["pagerank_099"])
    rho_050_099, _ = spearmanr(df_pr["pagerank_050"], df_pr["pagerank_099"])

    print(f"\nSpearman Rank Correlations Across Damping Factors:")
    print(f"  rho(d=0.50, d=0.85) = {rho_050_085:.4f}")
    print(f"  rho(d=0.85, d=0.99) = {rho_085_099:.4f}")
    print(f"  rho(d=0.50, d=0.99) = {rho_050_099:.4f}")

    # Top-10 Shift Table
    # Compare top nodes from d=0.85 across other damping factors
    damping_comparison = top10_pr085[["node", "rank_pr_085", "rank_pr_050", "rank_pr_099", "pagerank_085", "pagerank_050", "pagerank_099"]].copy()
    damping_comparison["shift_050_to_085"] = damping_comparison["rank_pr_050"] - damping_comparison["rank_pr_085"]
    damping_comparison["shift_085_to_099"] = damping_comparison["rank_pr_085"] - damping_comparison["rank_pr_099"]
    damping_shift_path = OUTPUTS_DIR / f"{dataset_key}_pagerank_damping_shifts.csv"
    damping_comparison.to_csv(damping_shift_path, index=False)

    # ============================================================
    # TASK 5: HITS ALGORITHM (HUBS & AUTHORITIES)
    # ============================================================
    print("\n--- Running Kleinberg HITS Algorithm ---")
    t0 = time.time()
    try:
        hubs, authorities = nx.hits(G, max_iter=2000, tol=1e-08, normalized=True)
    except nx.PowerIterationFailedConvergence:
        print("HITS power iteration with standard tol failed, falling back with numpy or relaxed tol...")
        hubs, authorities = nx.hits(G, max_iter=5000, tol=1e-06, normalized=True)
    hits_runtime = time.time() - t0
    print(f"[Done] HITS algorithm converged in {hits_runtime:.3f}s")

    df_hits = pd.DataFrame({
        "node": nodes,
        "hub_score": [hubs[n] for n in nodes],
        "authority_score": [authorities[n] for n in nodes]
    })
    df_hits["rank_hub"] = df_hits["hub_score"].rank(ascending=False, method="min").astype(int)
    df_hits["rank_authority"] = df_hits["authority_score"].rank(ascending=False, method="min").astype(int)
    df_hits["asymmetry"] = df_hits["hub_score"] - df_hits["authority_score"]

    hits_full_path = OUTPUTS_DIR / f"{dataset_key}_hits_all.csv"
    df_hits.to_csv(hits_full_path, index=False)

    # Top-10 Hubs
    top10_hubs = df_hits.sort_values(by="hub_score", ascending=False).head(10).reset_index(drop=True)
    top10_hubs_path = OUTPUTS_DIR / f"{dataset_key}_top10_hits_hubs.csv"
    top10_hubs[["node", "hub_score", "authority_score", "rank_authority"]].rename(
        columns={"node": "Node ID", "hub_score": "Hub Score", "authority_score": "Authority Score", "rank_authority": "Authority Rank"}
    ).to_csv(top10_hubs_path, index=False)

    print(f"\nTop-10 HITS Hubs:")
    for idx, row in top10_hubs.iterrows():
        print(f"  Rank {idx+1:2d}: {row['node']:<35s} | Hub: {row['hub_score']:.6f} | Auth: {row['authority_score']:.6f} (Rank {int(row['rank_authority']):2d})")

    # Top-10 Authorities
    top10_authorities = df_hits.sort_values(by="authority_score", ascending=False).head(10).reset_index(drop=True)
    top10_authorities_path = OUTPUTS_DIR / f"{dataset_key}_top10_hits_authorities.csv"
    top10_authorities[["node", "authority_score", "hub_score", "rank_hub"]].rename(
        columns={"node": "Node ID", "authority_score": "Authority Score", "hub_score": "Hub Score", "rank_hub": "Hub Rank"}
    ).to_csv(top10_authorities_path, index=False)

    print(f"\nTop-10 HITS Authorities:")
    for idx, row in top10_authorities.iterrows():
        print(f"  Rank {idx+1:2d}: {row['node']:<35s} | Auth: {row['authority_score']:.6f} | Hub: {row['hub_score']:.6f} (Rank {int(row['rank_hub']):2d})")

    # Structural Asymmetry Analysis:
    # 1. Pure Hubs: Nodes in top-20 Hubs with lowest Authority rank (high hub, low authority)
    pure_hubs = df_hits.sort_values(by="asymmetry", ascending=False).head(10)
    # 2. Pure Authorities: Nodes in top-20 Authorities with lowest Hub rank (high authority, low hub)
    pure_authorities = df_hits.sort_values(by="asymmetry", ascending=True).head(10)

    asymmetry_df = pd.concat([
        pure_hubs.assign(Category="Pure Hub (Broad Subscriber / Directory)"),
        pure_authorities.assign(Category="Pure Authority (Content Producer / Star)")
    ])
    asymmetry_path = OUTPUTS_DIR / f"{dataset_key}_hits_structural_asymmetry.csv"
    asymmetry_df.to_csv(asymmetry_path, index=False)

    print(f"\nStructural Asymmetry Identified (Saved to {asymmetry_path.name}):")
    print("  Top Pure Hubs (High Hub, Low Authority):")
    for _, row in pure_hubs.head(3).iterrows():
        print(f"    - {row['node']}: Hub={row['hub_score']:.6f}, Auth={row['authority_score']:.6f}")
    print("  Top Pure Authorities (High Authority, Low Hub):")
    for _, row in pure_authorities.head(3).iterrows():
        print(f"    - {row['node']}: Auth={row['authority_score']:.6f}, Hub={row['hub_score']:.6f}")

    return df_pr, df_hits, spearman_pr_indeg, damping_comparison

if __name__ == "__main__":
    compute_pagerank_and_hits()
